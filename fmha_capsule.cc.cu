#include "utils.h"
#include <cstddef>
#include <cstdio>
#include <mma.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

using namespace nvcuda;
namespace py = pybind11;

template <int Bx>
struct GemmBlockTile {
    const int Br, Bc;
    const int ntile_y = Br / 16;
    const int ntile_x = Bc / 16;
    const int num_warps = Bx / 32;
    const int tiles_per_warp = (ntile_y * ntile_x + num_warps - 1) / num_warps;
    const int num_loops = tiles_per_warp;
    const int loop_stride = num_warps;
    const int numbers_per_thread = Br*Bc/Bx;

    int loop_id;
    int tid, warp_id, lane_id;
    int warp_tile_y, warp_tile_x;
    int ldaA, ldaB, ldaBlock;
    float *warpA, *warpB, *warpC;
    float *blockA, *blockB, *blockC;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;

    float *load_reg;
    float *accum_reg;
    
    __device__ GemmBlockTile(float *blockA, float *blockB, float *blockC, int m, int n, int k):
    blockA(blockA), blockB(blockB), blockC(blockC), ldaA(k), ldaB(n), ldaBlock(n),
    tid(threadIdx.x), warp_id(threadIdx.x / 32), lane_id(threadIdx.x % 32), loop_id(0)
    {}

    __device__ inline void move_block_ab() {
        blockA += 8;
        blockB += 8 * ldaB;
    }

    __device__ inline void get_warp_abc() {
        warpA = blockA + 16 * ldaA * warp_tile_y;
        warpB = blockB + 16 * warp_tile_x;
        warpC = blockC + 16 * ldaBlock * warp_tile_y + 16 * warp_tile_x;
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

    __device__ inline void store() { wmma::store_matrix_sync(warpC, frag_c, ldaBlock, wmma::mem_row_major); }

    __device__ inline void load_to_accum() {

        #pragma unroll
        for (int i=0; i<numbers_per_thread; i+=2) {
            FetchFloat2(load_reg[i]) = FetchFloat2(blockC[i*Bx+tid*2]);
        }

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i)
            accum_reg[i] += load_reg[i];
    }

    __device__ inline void save_to_block_out() {
        #pragma unroll
        for (int i=0; i<numbers_per_thread; i+=2)
            FetchFloat2(blockC[i*Bx+tid*2]) = FetchFloat2(accum_reg[i]);
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

        float accum_reg_[numbers_per_thread];
        float load_reg_[numbers_per_thread];
        accum_reg = accum_reg_;
        load_reg = load_reg_;

        #pragma unroll
        for (int i=0; i<(ldaA/8); ++i) {
            loop_mma_store();
            load_to_accum();
            move_block_ab();
        }
        save_to_block_out();
        __syncthreads();
    }
};

// For Bc >= 128, Br = 128(?)

template <int N, int d, int Br, int Bc, int Bx>
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

template <int N, int d, int Br, int Bc, int Bx>
__device__ inline void update_o(
    float *block_o, float* smem_mat, float* smem_sum, float* smem_max, float* v_init,
    float *smem_buffer_pv
    ) {
    int tid = threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int n_warps = Bx / 32;
    int n_loops = Br / n_warps;
    const int number_per_thread = d / 32;
    int loop_stride = n_warps;
    
    GemmBlockTile<Br, d, Bx> gemm_pv(smem_mat, v_init, smem_buffer_pv, Bc, d);
    gemm_pv.calculate_gemm();
    /* printf("%.2f\n",smem_buffer_pv[2047]); */

    // Now calculate o
    
    float load_o_reg[number_per_thread];
    float load_pv_reg[number_per_thread];
    float li_new, li, mi, mi_new, m_ij;
    float o_out[number_per_thread];
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

template <int N, int d, int Br, int Bc, int Bx>
__device__ void FlashAttentionTile(
    float *q, float *k_init, float *v_init, float *o, float *global_sum, float *global_max,
    float *smem_buffer, float *smem_sum, float *smem_max, float *smem_buffer_pv
) {
    const int n_rtile = N / Br;
    const int r_tile = blockIdx.x;
    
    float *q_init = q + d * Br * r_tile;
    /* __shared__ float smem_buffer[Br*Bc]; */

    GemmBlockTile<Br, Bc, Bx> gemm(q_init, k_init, smem_buffer, d, N);
    gemm.calculate_gemm();
    
    /* __shared__ float smem_sum[Br*32]; */
    /* __shared__ float smem_max[Br*32]; */

    float *block_global_sum = global_sum + Br * r_tile;
    float *block_global_max = global_max + Br * r_tile;
    
    LocalSoftmax<N, d, Br, Bc, Bx>local_softmax (smem_buffer, smem_sum, smem_max, block_global_sum, block_global_max);
    local_softmax.softmax_loop();

    float * block_o = o + d * Br * r_tile;

    /* __shared__ float smem_buffer_pv[Br*d]; */
    update_o<N, d, Br, Bc, Bx>(block_o, smem_buffer, smem_sum, smem_max, v_init, smem_buffer_pv);
}

template <int N, int d, int Br, int Bc, int Bx>
__global__ void FlashAttentionKernel(float *q, float *k, float *v, float *global_sum, float *global_max, float *o) {
    int num_loops = d / Bc;
    float *k_init;
    float *v_init;

    __shared__ float smem_buffer[Br*Bc];
    __shared__ float smem_sum[Br*32];
    __shared__ float smem_max[Br*32];
    __shared__ float smem_buffer_pv[Br*d];

    #pragma unroll
    for (int tile_c = 0; tile_c<d; tile_c+=Bc) {
        k_init = k + tile_c;
        v_init = v + tile_c * d;
        FlashAttentionTile<N, d, Br, Bc, Bx>(q, k_init, v_init, o, global_sum, global_max, smem_buffer, smem_sum, smem_max, smem_buffer_pv);
    }
}


void jax_fhma(cudaStream_t stream, void** buffers, const char * opaque, size_t opaque_len) {
    printf("Now use custom XLA!\n");
    /* std::cout << opaque << std::endl; */
    float* q = reinterpret_cast<float*>(buffers[0]);
    float* k = reinterpret_cast<float*>(buffers[1]);
    float* v = reinterpret_cast<float*>(buffers[2]);
    float* o = reinterpret_cast<float*>(buffers[3]);

    char *op = new char [opaque_len];
    for (int i=0; i<opaque_len; ++i) {
        op[i] = opaque[i];
    }
    int *info = reinterpret_cast<int*>(op);
    // [N,d]

    int N = info[0];
    int d = info[1];
    int Br = 32;
    int Bc = 128;
    int Bx = 512;
    int Gx = N / Bx;

    float *global_max, *global_sum;
    size_t nbytesSum = sizeof(float) * Br;
    cudaMalloc(&global_max, nbytesSum);
    cudaMalloc(&global_sum, nbytesSum);
    cudaMemset(global_max, 0, nbytesSum);
    cudaMemset(global_sum, 0, nbytesSum);

    size_t nbytesSMEM = (Br*Bc+Br*32+Br*32+Br*d)*sizeof(float);

    FlashAttentionKernel<N, d, Br, Bc, Bx><<<Gx,Bx,nbytesSMEM,stream>>>(q, k, v, global_sum, global_max, o);
}

PYBIND11_MODULE(example, m) {
    m.doc() = "jax flash attention";
    m.def("fhma_jax", 
        /* & py_cuda_add, */
        [](){
        const char* name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *) &jax_fhma, name);
        },
        "HAHA! PY ADDED!"
    );
}
