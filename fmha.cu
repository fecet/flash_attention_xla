#include "utils.h"
#include <cstddef>
#include <cstdio>
#include <mma.h>

using namespace nvcuda;

template <const int Bx>
struct GemmBlockTile {

    int Br, Bc;
    int ntile_y; 
    int ntile_x; 
    int num_warps; 
    int tiles_per_warp; 
    int num_loops; 
    int loop_stride; 
    int numbers_per_thread; 

    int Bd;
    float *blockA, *blockB, *blockC;
    int ldaA, ldaB, ldaC;
    float *warpA, *warpB, *warpC;

    int loop_id;
    int tid, warp_id, lane_id;
    int warp_tile_y, warp_tile_x;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;

    float accum[2*16];
    
    __device__ GemmBlockTile(float* blockA, int ldaA, float *blockB, int ldaB, float *blockC, int ldaC, int Br, int Bc, int Bd):
    blockA(blockA), blockB(blockB), blockC(blockC), ldaA(ldaA), ldaB(ldaB), ldaC(ldaC), Br(Br), Bc(Bc), Bd(Bd),
    loop_id(0), tid(threadIdx.x), warp_id(threadIdx.x/32), lane_id(threadIdx.x%32)
    {
        ntile_y = Br / 16;
        ntile_x = Bc / 16;
        num_warps = Bx / 32;
        tiles_per_warp = (ntile_y * ntile_x + num_warps - 1) / num_warps;
        num_loops = tiles_per_warp;
        loop_stride = num_warps;
        numbers_per_thread = Br*Bc/Bx;
    }

    __device__ inline void move_block_ab() {
        blockA += 8;
        blockB += 8 * ldaB;
    }

    __device__ inline void get_warp_abc() {
        warpA = blockA + 16 * ldaA * warp_tile_y;
        warpB = blockB + 16 * warp_tile_x;
        warpC = blockC + 16 * ldaC * warp_tile_y + 16 * warp_tile_x;
    }

    __device__ inline void get_warp_tile_xy() {
        warp_tile_y = (loop_id * loop_stride + warp_id) / ntile_x;
        warp_tile_x = (loop_id * loop_stride + warp_id) % ntile_x;
    }

    __device__ inline void fill_load() {
        wmma::fill_fragment(frag_c, 0.0f);
        wmma::load_matrix_sync(frag_a, warpA, ldaA);
        wmma::load_matrix_sync(frag_b, warpB, ldaB);
    }
    
    __device__ inline void mma() { wmma::mma_sync(frag_c, frag_a, frag_b, frag_c); }

    __device__ inline void store() { wmma::store_matrix_sync(warpC, frag_c, ldaC, wmma::mem_row_major); }
    
    __device__ inline void load_to_accum() {

        #pragma unroll
        for (int i=0; i<numbers_per_thread; i+=2) {
            FetchFloat2(accum[numbers_per_thread+i]) = FetchFloat2(
            blockC[LdIdx((Bx*i+tid*2)/Bc,(Bx*i+tid*2)%Bc,ldaC)]
            );
        }

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i)
            accum[i] += accum[numbers_per_thread+i];
    }

    __device__ inline void save_to_block_out() {
        #pragma unroll
        for (int i=0; i<numbers_per_thread; i+=2)
            FetchFloat2(
                blockC[LdIdx((Bx*i+tid*2)/Bc,(Bx*i+tid*2)%Bc,ldaC)]
            ) = FetchFloat2(accum[i]);
    }

    __device__ inline void loop_mma_store() {
        loop_id = 0;

        #pragma unroll
        for (int i=0; i<num_loops; ++i) {
            get_warp_tile_xy();
            get_warp_abc();
            fill_load();
            mma();
            store();
            loop_id ++;
        }
        __syncthreads();
    }

    __device__ inline void calculate_gemm() {

        #pragma unroll
        for (int i=0; i<(Bd/8); ++i) {
            loop_mma_store();
            load_to_accum();
            move_block_ab();
        }
        save_to_block_out();
        __syncthreads();
    }
};

// For Bc >= 128, Br = 128(?)

template <const int Br, const int Bc, const int Bx>
struct LocalSoftmax {
    // One warp per row
    // Using float4
    static const int num_per_thread = Bc / 32;
    static const int num_warps = Bx / 32;
    static const int num_loops = Br / num_warps;
    static const int loop_stride = num_warps;

    int loop_id = 0; // for loop_id in range(num_loops)
    float *smem_mat; 
    float *gmem_sum, *gmem_max;
    float *smem_sum, *smem_max; //sum and max [Br * 32]

    float thread_fetch_reg[num_per_thread];
    float thread_fx[num_per_thread];
    float thread_m, thread_l;
    float m_ij, l_ij;

    int warp_id, lane_id, tid;

    __device__ LocalSoftmax(
        float *smem_mat, float* smem_sum, float* smem_max,
        float *gmem_sum, float* gmem_max // gmem_max and gemm_sum pointed at the first row of the block
    ):
    smem_mat(smem_mat), smem_max(smem_max), smem_sum(smem_sum),
    tid(threadIdx.x), warp_id(threadIdx.x/32), lane_id(threadIdx.x%32),
    gmem_max(gmem_max), gmem_sum(gmem_sum)
    {}

    __device__ inline void fetch_to_reg() {
        /* #pragma unroll */
        /* for (int i=0,j=0; i<num_per_thread; i+=4,++j) */
        /*     FetchFloat4(thread_fetch_reg[i]) = FetchFloat4( */
        /*         smem_mat[LdIdx(loop_id*loop_stride+warp_id, lane_id*4+j*128, Bc)] */
        /*     ); */
        #pragma unroll
        for (int i=0; i<num_per_thread; ++i)
            thread_fetch_reg[i] = smem_mat[LdIdx(loop_id*loop_stride+warp_id, lane_id+i*32, Bc)];
    }

    __device__ inline void thread_max() {
        thread_m = thread_fetch_reg[0];
        for (int i=1; i<num_per_thread; ++i) 
            if (thread_fetch_reg[i] > thread_m)
                thread_m = thread_fetch_reg[i];
    }

    __device__ inline void block_max() {
        smem_max[loop_id*loop_stride*32+warp_id*32+lane_id] = thread_m;
        __syncthreads();
        if (lane_id == 0) {
            float temp_reg[32];

            #pragma unroll
            for (int i=0; i<32; i+=4)
                FetchFloat4(temp_reg[i]) = FetchFloat4(smem_max[loop_id*loop_stride*32+warp_id*32+i]);

            float block_max = temp_reg[0];
            for (int i=1; i<32; ++i)
                if (temp_reg[i]>block_max)
                    block_max = temp_reg[i];
            smem_max[loop_id*loop_stride*32+warp_id*32+0] = block_max;
        }
        __syncthreads();
    }

    __device__ inline void get_block_max() { m_ij = smem_max[loop_id*loop_stride*32+warp_id*32+0]; }

    __device__ inline void thread_softmax() {
        // After get block_m
        thread_l = 0.0f;
        #pragma unroll
        for (int i=0; i<num_per_thread; ++i) {
            thread_fx[i] = expf(thread_fetch_reg[i]-m_ij);
            thread_l += thread_fx[i];
        }
    }

    __device__ inline void save_mat_to_smem() {
        /* #pragma unroll */
        /* for (int i=0,j=0; i<num_per_thread; i+=4,++j) */
        /*     FetchFloat4( */
        /*         smem_mat[LdIdx(loop_id*loop_stride+warp_id, lane_id*4+j*128, Bc)] */
        /*     ) = FetchFloat4(thread_fx[i]); */
        #pragma unroll
        for (int i=0; i<num_per_thread; ++i)
            smem_mat[LdIdx(loop_id*loop_stride+warp_id, lane_id+i*32, Bc)] = thread_fx[i];

    }

    __device__ inline void block_sum() {
        smem_sum[loop_id*loop_stride*32+warp_id*32+lane_id] = thread_l;
        __syncthreads();
        if (lane_id == 0) {
            float temp_reg[32];

            #pragma unroll
            for (int i=0; i<32; i+=4)
                FetchFloat4(temp_reg[i]) = FetchFloat4(smem_sum[loop_id*loop_stride*32+warp_id*32+i]);

            float block_sum = 0.0f;
            for (int i=1; i<32; ++i)
                block_sum += temp_reg[i];
            smem_sum[loop_id*loop_stride*32+warp_id*32+0] = block_sum;
        }
        __syncthreads(); // need ?
    }

    __device__ inline void get_block_sum() { l_ij = smem_sum[loop_id*loop_stride*32+warp_id*32+0]; }

    __device__ inline void get_ml_new() {
        if (lane_id == 0) {
            float mi = gmem_max[loop_id*loop_stride+warp_id];
            float li = gmem_sum[loop_id*loop_stride+warp_id];
            float m_new = mi > m_ij ? mi : m_ij;
            float l_new = expf(mi-m_new)*li+expf(m_ij-m_new)*l_ij;
            
            smem_max[loop_id*loop_stride*32+warp_id*32+2] = mi;
            smem_sum[loop_id*loop_stride*32+warp_id*32+2] = li;

            gmem_max[loop_id*loop_stride+warp_id] = m_new;
            gmem_sum[loop_id*loop_stride+warp_id] = l_new;

            smem_max[loop_id*loop_stride*32+warp_id*32+1] = m_new;
            smem_sum[loop_id*loop_stride*32+warp_id*32+1] = l_new;
        }
        // smem_max,smem_sum [0]: mij, [2]: mi_last,li_last aka mi,li [1]: mi_new, li_new
    }

    __device__ inline void softmax_loop() {
        loop_id = 0;
        #pragma unroll
        for (int l=0; l<Br; l+=loop_stride,++loop_id) {
            fetch_to_reg();
            thread_max();
            block_max();
            get_block_max();
            thread_softmax();
            save_mat_to_smem();
            block_sum();
            get_block_sum();
            get_ml_new();
            __syncthreads();
        }
    }
};

template <const int Br, const int Bc, const int Bx>
__device__ inline void update_o(
    float *block_o, float* smem_mat, float* smem_sum, float* smem_max, float* v_init,
    float *smem_buffer_pv,
    const int N, const int d
    ) {
    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int n_warps = Bx / 32;
    int n_loops = Br / n_warps;
    const int number_per_thread = d / 32;
    int loop_stride = n_warps;
    
    /* GemmBlockTile<Bx> gemm_pv(smem_mat, v_init, smem_buffer_pv, Bc, d); */
    GemmBlockTile<Bx> gemm_pv(smem_mat, Bc, v_init, d, smem_buffer_pv, d, Br, d, Bc);
    gemm_pv.calculate_gemm();
    /* printf("%.2f\n",smem_buffer_pv[2047]); */

    // Now calculate o
    
    float load_o_reg[8]; // 128 / 32
    float load_pv_reg[8]; // 128 / 32
    float li_new, li, mi, mi_new, m_ij;
    float o_out[8]; // 128 /32
    for (int loop_id=0; loop_id<n_loops; ++loop_id) {
        // Load
        #pragma unroll
        for (int i=0,j=0; i<number_per_thread; i+=2,++j) {
            FetchFloat2(load_o_reg[i]) = FetchFloat2(block_o[LdIdx(loop_id*loop_stride+warp_id, j*64+lane_id*2, d)]);
            FetchFloat2(load_pv_reg[i]) = FetchFloat2(smem_buffer_pv[LdIdx(loop_id*loop_stride+warp_id, j*64+lane_id*2, d)]);
        }

        int row_id = loop_id*loop_stride + warp_id;
        li_new = smem_sum[row_id*32+1];
        mi_new = smem_max[row_id*32+1];
        li = smem_sum[row_id*32+2];
        mi = smem_max[row_id*32+2];
        m_ij = smem_max[row_id*32+0];
        /* printf("%.2f, %.2f\n",m_ij,mi_new); */
        
        // Compute
        #pragma unroll
        for (int i=0; i<number_per_thread; ++i) {
            o_out[i] = (1/li_new)*((li*expf(mi-mi_new)*load_o_reg[i])+expf(m_ij-mi_new)*load_pv_reg[i]);
            /* printf("%.2f\n",expf(m_ij-mi_new)); */
        }

        //Save
        #pragma unroll
        for (int i=0,j=0; i<number_per_thread; i+=2,++j) 
            FetchFloat2(block_o[LdIdx(loop_id*loop_stride+warp_id, j*64+lane_id*2, d)]) = FetchFloat2(o_out[i]);
    }
    
}

template <const int Br, const int Bc, const int Bx>
__device__ void FlashAttentionTile(
    float *q, float *k_init, float *v_init, float *o, float *global_sum, float *global_max,
    float *smem_buffer, float *smem_sum, float *smem_max, float *smem_buffer_pv,
    const int N, const int d
) {
    const int n_rtile = N / Br;
    const int r_tile = blockIdx.x;
    
    float *q_init = q + d * Br * r_tile;
    /* __shared__ float smem_buffer[Br*Bc]; */

    GemmBlockTile<Bx> gemm(q_init, d, k_init, N, smem_buffer, Bc, Br, Bc, d);
    gemm.calculate_gemm();
    
    /* __shared__ float smem_sum[Br*32]; */
    /* __shared__ float smem_max[Br*32]; */

    float *block_global_sum = global_sum + Br * r_tile;
    float *block_global_max = global_max + Br * r_tile;
    
    LocalSoftmax<Br, Bc, Bx>local_softmax (smem_buffer, smem_sum, smem_max, block_global_sum, block_global_max);
    local_softmax.softmax_loop();

    float * block_o = o + d * Br * r_tile;

    /* __shared__ float smem_buffer_pv[Br*d]; */
    update_o<Br, Bc, Bx>(block_o, smem_buffer, smem_sum, smem_max, v_init, smem_buffer_pv, N, d);
}

template <const int Br, const int Bc, const int Bx>
__global__ void FlashAttentionKernel(
    float *q, float *k, float *v, float *global_sum, float *global_max, float *o,
    const int N, const int d
) {
    int num_loops = d / Bc;
    float *k_init;
    float *v_init;

    __shared__ float smem_buffer[Br*Bc];
    __shared__ float smem_sum[Br*32];
    __shared__ float smem_max[Br*32];
    __shared__ float smem_buffer_pv[Br*128];

    for (int tile_c = 0; tile_c<d; tile_c+=Bc) {
        k_init = k + tile_c;
        v_init = v + tile_c * d;
        FlashAttentionTile<Br, Bc, Bx>(q, k_init, v_init, o, global_sum, global_max, smem_buffer, smem_sum, smem_max, smem_buffer_pv, N, d);
    }
}

const int N = 8192*2;
const int d = 128;
const int Br = 32;
const int Bc = 128;
const int Bx = 512;
const int Gx = N / Bx;

int main() {
    float *q = new float [N*d];
    float *k = new float [d*N];
    float *v = new float [N*d];
    float *out = new float [N*d];
    
    for (int i=0; i<N*d; ++i) q[i] = 2.0f;
    for (int i=0; i<N*d; ++i) k[i] = 2.0f;
    for (int i=0; i<N*d; ++i) v[i] = 2.0f;
    for (int i=0; i<N*d; ++i) out[i] = 0.0f;

    cudaError_t stat;
    float *dq, *dk, *dv, *d_out;
    float *global_max, *global_sum;
    size_t nbytesQKV = sizeof(float) * N * d;
    size_t nbytesSum = sizeof(float) * Br;

    stat = cudaMalloc(&dq, nbytesQKV);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&dk, nbytesQKV);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&dv, nbytesQKV);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&d_out, nbytesQKV);
    CUDACheck(stat, "malloc");
    
    
    stat = cudaMalloc(&global_max, nbytesSum);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&global_sum, nbytesSum);
    CUDACheck(stat, "malloc");

    stat = cudaMemcpy(dq, q, nbytesQKV, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(dk, k, nbytesQKV, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(dv, v, nbytesQKV, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(d_out, out, nbytesQKV, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    
    stat = cudaMemset(global_max, 0, nbytesSum);
    CUDACheck(stat, "memset");
    stat = cudaMemset(global_sum, 0, nbytesSum);
    CUDACheck(stat, "memset");
    
TimerInit
TimerStart
    FlashAttentionKernel<Br ,Bc, Bx><<<Gx,Bx>>>(dq, dk, dv, global_sum, global_max, d_out, N, d);
    cudaDeviceSynchronize();
    stat = cudaGetLastError();
    CUDACheck(stat, "kernel");
    cudaMemcpy(out, d_out, nbytesQKV, cudaMemcpyDeviceToHost);
TimerEnd("KK")
    for (int i=0; i<N*d; ++i) {
        printf("%.2f ", out[i]);
    }
    return 0;
}


