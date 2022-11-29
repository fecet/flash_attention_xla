/* #include <__clang_cuda_builtin_vars.h> */
#include "utils.h"
#include <cstdio>
#include <cstdlib>
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
    float accum[32];

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
        
        /* accum = accum_defined_out; */
        
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


const int M = 128;
const int N = 128;
const int K = 1024;

const int Br = 32;
const int Bc = 32;
const int Bx = 64;

__global__ void gemm_tc_kernel(float* a, float*b, float* c){
    const int ldaA = K;
    const int ldaB = N;
    const int ldaC = N;
    const int Bd = K;
    float* blockA = a + blockIdx.y * Br * ldaA;
    float* blockB = b + blockIdx.x * Bc;
    float* blockC = c + blockIdx.y * Br * ldaC + blockIdx.x * Bc;

    
    /* __shared__ float smem_buffer[Bm*Bn]; */
    GemmBlockTile<Bx> gemm(blockA, ldaA, blockB, ldaB, blockC, ldaC, Br, Bc, Bd);
    /* float accum[2*Br*Bc/Bx]; */
    gemm.calculate_gemm();

    /* #pragma unroll */
    /* for (int i=0,j=0; i<gemm.num_per_thread; i+=4,++j) */
    /*     FetchFloat4(c[j*gemm.tid*4]) = FetchFloat4(smem_buffer[j*gemm.tid*4]); */
}

int main() {
    float *a = new float[M*K];
    float *b = new float[K*N];
    float *c = new float[M*N];

    for (int i=0; i<M*K; ++i) a[i] = (float)rand() / RAND_MAX;
    for (int i=0; i<N*K; ++i) b[i] = (float)rand() / RAND_MAX;
    /* for (int i=0; i<M*K; ++i) a[i] = i % 13; */
    /* for (int i=0; i<N*K; ++i) b[i] = i % 13; */
    for (int i=0; i<M*N; ++i) c[i] = 0.0f;

    float *da, *db, *dc;
    size_t nbytesA = sizeof(float) * M * K;
    size_t nbytesB = sizeof(float) * K * N;
    size_t nbytesC = sizeof(float) * M * N;

    cudaError_t stat;
    stat = cudaMalloc(&da, nbytesA);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&db, nbytesB);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&dc, nbytesC);
    CUDACheck(stat, "malloc");

    stat = cudaMemcpy(da, a, nbytesA, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(db, b, nbytesB, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(dc, c, nbytesC, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    
    dim3 grid_dim;
    grid_dim.y = M / Br;
    grid_dim.x = N / Bc;
    gemm_tc_kernel<<<grid_dim,Bx>>>(da, db, dc);
    cudaDeviceSynchronize();
    stat = cudaGetLastError();
    CUDACheck(stat, "kernel");

    cudaMemcpy(c, dc, nbytesC, cudaMemcpyDeviceToHost);
    CUDACheck(stat, "memcpy_back");

    cudaFree(dc);

    float *blas_c = new float [M*N];
    float *blas_dc;
    stat = cudaMalloc(&blas_dc, nbytesC);
    CUDACheck(stat, "blas_dc malloc");
    for (int i=0; i<M*N; ++i) blas_c[i] = 0.0f;

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cublasStatus_t blas_stat;
    float alpha = 1.0;
    float beta = 0;
    stat = cudaMemcpy(blas_dc, blas_c, nbytesC, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy blas t")
    blas_stat = cublasSgemm_v2(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, N, da, K, &beta, blas_dc, N);
    CUBLASCheck(blas_stat, "cublas");
    cudaMemcpy(blas_c, blas_dc, nbytesC, cudaMemcpyDeviceToHost);
    float eps = 0.01;
    for (int i=0; i<M*N; ++i) {
        float delta = c[i]-blas_c[i];
        if (delta > eps || -delta>eps) {
            printf("%d my: %.4f, blas: %.4f ",i,c[i],blas_c[i]);
            printf("delta: %.4f\n",c[i]-blas_c[i]);
        }
    }
    
    return 0;
}
