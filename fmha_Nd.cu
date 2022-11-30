#include "utils.h"
#include <cstdio>
#include <mma.h>
#include <cmath>
/* #include <pybind11/pybind11.h> */
/* #include <pybind11/numpy.h> */
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

using namespace nvcuda;
/* namespace py = pybind11; */

template <const int Bx>
struct GemmBlockTilePV {

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
    float accum[32] = {0.0f};

    int loop_id;
    int tid, warp_id, lane_id;
    int warp_tile_y, warp_tile_x;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;
    
    __device__ GemmBlockTilePV(float* blockA, int ldaA, float *blockB, int ldaB, float *blockC, int ldaC, int Br, int Bc, int Bd):
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

        /* #pragma unroll */
        /* for (int i=0; i<numbers_per_thread; i+=2) { */
        /*     FetchFloat2(accum[numbers_per_thread+i]) = FetchFloat2( */
        /*     blockC[LdIdx((Bx*i+tid*2)/Bc,(Bx*i+tid*2)%Bc,ldaC)] */
        /*     ); */
        /* } */
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            accum[numbers_per_thread+i] = blockC[LdIdx((Bx*i+tid)/Bc, (Bx*i+tid)%Bc, ldaC)];
            /* accum[numbers_per_thread+i] = blockC[1024*128]; */
        }

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i)
            accum[i] += accum[numbers_per_thread+i];
    }

    __device__ inline void save_to_block_out() {
        /* #pragma unroll */
        /* for (int i=0; i<numbers_per_thread; i+=2) */
        /*     FetchFloat2( */
        /*         blockC[LdIdx((Bx*i+tid*2)/Bc,(Bx*i+tid*2)%Bc,ldaC)] */
        /*     ) = FetchFloat2(accum[i]); */
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            blockC[LdIdx((Bx*i+tid)/Bc, (Bx+i+tid)%Bc, ldaC)] = accum[i];
        }
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
            if (i != 0) {
                move_block_ab();
            }
            loop_mma_store();
            load_to_accum();
        }
        save_to_block_out();
        __syncthreads();
    }
};

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
    float accum[32] = {0.0f};

    int loop_id;
    int tid, warp_id, lane_id;
    int warp_tile_y, warp_tile_x;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;
    
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

        /* #pragma unroll */
        /* for (int i=0; i<numbers_per_thread; i+=2) { */
        /*     FetchFloat2(accum[numbers_per_thread+i]) = FetchFloat2( */
        /*     blockC[LdIdx((Bx*i+tid*2)/Bc,(Bx*i+tid*2)%Bc,ldaC)] */
        /*     ); */
        /* } */
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            accum[numbers_per_thread+i] = blockC[LdIdx((Bx*i+tid)/Bc, (Bx+i+tid)%Bc, ldaC)];
        }

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i)
            accum[i] += accum[numbers_per_thread+i];
    }

    __device__ inline void save_to_block_out() {
        /* #pragma unroll */
        /* for (int i=0; i<numbers_per_thread; i+=2) */
        /*     FetchFloat2( */
        /*         blockC[LdIdx((Bx*i+tid*2)/Bc,(Bx*i+tid*2)%Bc,ldaC)] */
        /*     ) = FetchFloat2(accum[i]); */
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            blockC[LdIdx((Bx*i+tid)/Bc, (Bx+i+tid)%Bc, ldaC)] = accum[i];
        }
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
            if (i != 0) {
                move_block_ab();
            }
            loop_mma_store();
            load_to_accum();
        }
        save_to_block_out();
        __syncthreads();
    }
};

template <const int Br, const int Bc, const int Bx>
struct LocalSoftmax {
    static const int numbers_per_thread = Bc / 32;
    static const int num_warps = Bx / 32;
    static const int loop_stride = num_warps;
    static const int num_loops = Br / num_warps;

    float *smem_mat, *smem_reduce; // smem_reduce: (Br*32)
    float *gmem_sum, *gmem_max;
    
    float thread_m, thread_l;
    float thread_mat_reg[numbers_per_thread];
    float thread_fx[numbers_per_thread];

    float m_ij, l_ij;
    int warp_id, lane_id, tid;
    int loop_id;

    __device__ LocalSoftmax (
        float *smem_mat, float *smem_reduce, float *gmem_sum, float *gmem_max
    ): 
    smem_mat(smem_mat), smem_reduce(smem_reduce), gmem_sum(gmem_sum), gmem_max(gmem_max),
    tid(threadIdx.x), warp_id(threadIdx.x/32), lane_id(threadIdx.x%32), loop_id(0)
    {}

    __device__ inline void load_mat_to_reg() {
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            thread_mat_reg[i] = smem_mat[
                LdIdx(loop_id*loop_stride+warp_id,32*i+lane_id,Bc)
            ];
        }
    }

    __device__ inline void thread_max() {
        thread_m = thread_mat_reg[0];
        #pragma unroll
        for (int i=1; i<numbers_per_thread; ++i){
            if (thread_mat_reg[i] > thread_m) {
                thread_m = thread_mat_reg[i];
            }
        }
    }

    __device__ inline void block_max() {
        smem_reduce[(loop_id*loop_stride+warp_id)*32+lane_id] = thread_m;
        __syncthreads();
        if (lane_id == 0) {
            float max_reg[32];
            float block_m;

            #pragma unroll
            for (int i=0; i<32; i+=4) {
                FetchFloat4(max_reg[i]) = FetchFloat4(
                    smem_reduce[(loop_id*loop_stride+warp_id)*32+i]
                );
            }

            block_m = max_reg[0];

            #pragma unroll
            for (int i=1; i<32; ++i) {
                if (max_reg[i] > block_m)
                    block_m = max_reg[i];
            }
            smem_reduce[(loop_id*loop_stride+warp_id)*32+0] = block_m;
        }
        __syncthreads();
        m_ij = smem_reduce[(loop_id*loop_stride+warp_id)*32+0];
    }

    __device__ inline void thread_softmax() {
        thread_l = 0.0f;

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            thread_fx[i] = expf(thread_mat_reg[i]-m_ij);
            thread_l += thread_fx[i];
        }
    }


    __device__ inline void block_sum() {
        smem_reduce[(loop_id*loop_stride+warp_id)*32+lane_id] = thread_l;
        __syncthreads();
        if (lane_id == 0) {
            float sum_reg[32];
            float bsum = 0.0f;

            #pragma unroll
            for (int i=0; i<32; i+=4) {
                FetchFloat4(sum_reg[i]) = FetchFloat4(
                    smem_reduce[(loop_id*loop_stride+warp_id)*32+i]
                );
            }

            #pragma unroll
            for (int i=0; i<32; ++i) {
                bsum += sum_reg[i];
            }

            smem_reduce[(loop_id*loop_stride+warp_id)*32+0] = bsum;
        }
        __syncthreads();
        l_ij = smem_reduce[(loop_id*loop_stride+warp_id)*32+0];
    }

    __device__ inline void save_local_result_to_smem() {
        /* smem_reduce[(loop_id*loop_stride+warp_id)] */
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            smem_mat[LdIdx(loop_id*loop_stride+warp_id,32*i+lane_id,Bc)] = thread_fx[i];
        }
        if (lane_id == 0) {
            float mi = gmem_max[loop_id*loop_stride+warp_id];
            float li = gmem_sum[loop_id*loop_stride+warp_id];
            float mi_new = mi > m_ij ? mi : m_ij;
            float li_new = expf(mi-mi_new)*li+expf(m_ij-mi_new)*l_ij;

            int row_id = (loop_id*loop_stride+warp_id)*32;
            smem_reduce[row_id+0] = mi_new;
            smem_reduce[row_id+1] = li_new;
            smem_reduce[row_id+2] = mi;
            smem_reduce[row_id+3] = li;
            smem_reduce[row_id+4] = m_ij;

            gmem_max[loop_id*loop_stride+warp_id] = mi_new;
            gmem_sum[loop_id*loop_stride+warp_id] = li_new;
        }
        __syncthreads();
    }

    __device__ inline void softmax_loop() {
        loop_id = 0;
        #pragma unroll
        for (int i=0; i<Br; i+=loop_stride,++loop_id) {
            load_mat_to_reg();
            thread_max();
            block_max();
            thread_softmax();
            block_sum();
            save_local_result_to_smem();
        }
    }
};

template <const int Br, const int Bc, const int Bx>
__device__ inline void update_o(
    float *block_o, float *smem_mat, float *smem_reduce,
    float *v_init, float *smem_buffer_pv,
    int N, int d, int ldaPV
) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int num_warps = Bx / 32;
    int num_loops = Br / num_warps;
    int numbers_per_thread = d / 32;
    int loop_stride = num_warps;

    /* float temp = smem_buffer_pv[64*128]; */
    /* printf("%.2f\n",temp); */
    
    GemmBlockTilePV<Bx> gemm_pv(smem_mat, Bc, v_init, d, smem_buffer_pv, ldaPV, Br, d, Bc);
    gemm_pv.calculate_gemm();
    printf("HERE\n");
    
    float load_o_reg[8];
    float load_pv_reg[8];
    float li_new, li, mi, mi_new, m_ij;
    float o_out[8];


    #pragma unroll
    for (int loop_id=0; loop_id<num_loops; ++loop_id) {

        int row_id = (loop_id * loop_stride + warp_id)*32;
        mi_new = smem_reduce[row_id+0];
        li_new = smem_reduce[row_id+1];
        mi = smem_reduce[row_id+2];
        li = smem_reduce[row_id+3];
        m_ij = smem_reduce[row_id+4];

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            load_o_reg[i] = block_o[LdIdx(loop_id*loop_stride+warp_id, i*32+lane_id, d)];
            load_pv_reg[i] = smem_buffer_pv[LdIdx(loop_id*loop_stride+warp_id, i*32+lane_id, d)];
            o_out[i] =  (1/li_new)*((li*expf(mi-mi_new)*load_o_reg[i])+(expf(m_ij-mi_new)*load_pv_reg[i]));
            block_o[LdIdx(loop_id*loop_stride+warp_id, i*32+lane_id, d)] = o_out[i];
            /* printf("%.2f\n",o_out[i]); */
        }
    }
}

template <const int Br, const int Bc, const int Bx>
__device__ inline void FlashAttentionTile(
    float* q, float *k_init, float *v_init, float *o, float *global_sum, float *global_max,
    float *smem_buffer, float *smem_pv_buffer, float *smem_reduce, 
    int N, int d, int ldaPV
) {
    const int n_rtile = N / Br;
    const int r_tile = blockIdx.x;
    
    float *q_init = q + d * Br * r_tile;
    float *block_gsum = global_sum + r_tile * Br;
    float *block_gmax = global_max + r_tile * Br;

    float *block_o = o + d * Br * r_tile;

    GemmBlockTile<Bx>(q_init, d, k_init, N, smem_buffer, Bc, Br, Bc, d).calculate_gemm();
    LocalSoftmax<Br, Bc, Bx>(smem_buffer, smem_reduce, block_gsum, block_gmax).softmax_loop();
    update_o<Br, Bc, Bx>(block_o, smem_buffer, smem_reduce, v_init, smem_pv_buffer, N, d, ldaPV);
}

template <const int Br, const int Bc, const int Bx>
__global__ void FlashAttentionKernel(
    float *q, float *k, float *v, float *o, 
    float *global_sum, float *global_max,
    int N, int d
) {

    int num_loops = d / Bc;
    float *k_init;
    float *v_init;

    /* printf("%d\n",num_loops); */

    __shared__ float smem_buffer[Br*Bc];
    __shared__ float smem_reduce[Br*32];
    __shared__ float smem_buffer_pv[Br*128];

    #pragma unroll
    for (int tile_c = 0; tile_c<d; tile_c+=Bc) {
        k_init = k + tile_c;
        v_init = v + tile_c * d;
        FlashAttentionTile<Br, Bc, Bx>(q, k_init, v_init, o, global_sum, global_max, smem_buffer, smem_buffer_pv, smem_reduce, N, d, 128);
    }
}


const int Br = 32;
const int Bc = 32;
const int Bx = 128;

void fmha(cudaStream_t stream, void** buffers, const char * opaque, size_t opaque_len) {
    cudaError_t stat;
    std::cout << "CUSTOM XLAMBDA" << std::endl;
    float *q = reinterpret_cast<float*>(buffers[0]);
    float *k = reinterpret_cast<float*>(buffers[1]);
    float *v = reinterpret_cast<float*>(buffers[2]);
    float *o = reinterpret_cast<float*>(buffers[3]);
    
    int N,d;
    std::memcpy(&N, opaque, 4);
    std::memcpy(&d, opaque+4, 4);
    
    /* std::cout << d << std::endl; */

    float *gmem_sum = new float [N];
    float *gmem_max = new float [N];
    
    for (int i=0; i<N; ++i) {
        gmem_sum[i] = 0.0f;
        gmem_max[i] = -INFINITY;
    }
    
    float *d_gmem_sum, *d_gmem_max;
    stat = cudaMalloc(&d_gmem_sum, sizeof(float)*N);
    CUDACPPCheck(stat, "malloc")
    stat = cudaMalloc(&d_gmem_max, sizeof(float)*N);
    CUDACPPCheck(stat, "malloc")
    stat = cudaMemcpy(d_gmem_max, gmem_max, sizeof(float)*N, cudaMemcpyHostToDevice);
    CUDACPPCheck(stat, "memcpy")
    stat = cudaMemcpy(d_gmem_sum, gmem_sum, sizeof(float)*N, cudaMemcpyHostToDevice);
    CUDACPPCheck(stat, "memcpy")
    
    int Gx = N / Br;
    size_t nbytesSMEM = sizeof(float)*(Br*Bc+Br*32+Br*128);
    FlashAttentionKernel<Br, Bc, Bx><<<Gx,Bx,nbytesSMEM,stream>>>(q, k, v, o, d_gmem_sum, d_gmem_max, N, d);
    stat = cudaGetLastError();
    CUDACPPCheck(stat, "kernel")
    cudaDeviceSynchronize();
}

/* PYBIND11_MODULE(fmha, m) { */
/*     m.doc() = "SUSTensorTest of Kenel"; */
/*     m.def("fmha",  */
/*         [](){ */
/*         const char* name = "xla._CUSTOM_CALL_TARGET"; */
/*         return py::capsule((void *) &fmha, name); */
/*         }, */
/*         "Test of fmha" */
/*     ); */
/* } */


const int N = 128;
const int d = 128;
/* const int Br = 32; */
/* const int Bc = 64; */
/* const int Bx = 256; */
const int Gx = N / Br;

int main() {
    float *q = new float [N*d];
    float *k = new float [d*N];
    float *v = new float [N*d];
    float *out = new float [N*d];
    
    for (int i=0; i<N*d; ++i) q[i] = 3.0f;
    for (int i=0; i<N*d; ++i) k[i] = 3.0f;
    for (int i=0; i<N*d; ++i) v[i] = 3.0f;
    for (int i=0; i<N*d; ++i) out[i] = 0.0f;

    cudaError_t stat;
    float *dq, *dk, *dv, *d_out;
    float *global_max, *global_sum;
    size_t nbytesQKV = sizeof(float) * N * d;
    size_t nbytesSum = sizeof(float) * N;

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
    FlashAttentionKernel<Br ,Bc, Bx><<<Gx,Bx>>>(dq, dk, dv,d_out, global_sum, global_max, N, d);
    cudaDeviceSynchronize();
    stat = cudaGetLastError();
    CUDACheck(stat, "kernel");
    cudaMemcpy(out, d_out, nbytesQKV, cudaMemcpyDeviceToHost);
TimerEnd("KK")
    /* for (int i=0; i<N*d; ++i) { */
    /*     printf("%.2f ", out[i]); */
    /* } */
    return 0;
}
