/* #include <__clang_cuda_builtin_vars.h> */
#include <cstdio>
#include <mma.h>
#include "utils.h"

using namespace nvcuda;

template <const int Bm, const int Bn, const int Bx>
struct GemmBlockTile {
    static const int num_per_thread = Bm*Bn/Bx;
    int warp_id, lane_id, tid;
    int warp_tile_y, warp_tile_x;
    int ntile_y, ntile_x;
    float* warpA, *warpB;
    int ldaA, ldaB, ldaBlock;
    float* warp_smem_buffer;
    float* block_smem_buffer;

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;

    float accum[2][Bm*Bn/Bx] = {0.0f};

    __device__ GemmBlockTile(
        float* blockA, float* blockB, float* smem_buffer,
        int ldaA, int ldaB
    ):  ldaA(ldaA), ldaB(ldaB), ldaBlock(Bn), tid(threadIdx.x), warp_id(threadIdx.x/32),
        lane_id(threadIdx.x%32), ntile_y(Bm/16), ntile_x(Bn/16), block_smem_buffer(smem_buffer)
    {
        warp_tile_y = warp_id / ntile_x;
        warp_tile_x = warp_id % ntile_x;
        warpA = blockA + 16 * ldaA * warp_tile_y;
        warpB = blockB + 16 * warp_tile_x;
        warp_smem_buffer = smem_buffer + 16 * ldaBlock * warp_tile_y + 16 * warp_tile_x;
    }
    
    __device__ inline void fill_load() {
        wmma::fill_fragment(frag_c, 0.0f);
        wmma::load_matrix_sync(frag_a, warpA, ldaA);
        wmma::load_matrix_sync(frag_b, warpB, ldaB);
    }
    
    __device__ inline void move_ab() {
        warpA += 8;
        warpB += 8 * ldaB;
    }
    
    __device__ inline void mma() {
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    
    __device__ inline void save_buffer() {
        wmma::store_matrix_sync(warp_smem_buffer, frag_c, ldaBlock, wmma::mem_row_major);
    }
    
    __device__ inline void load_to_accum() {
        /* FetchFloat4(accum[1][0]) = FetchFloat4(block_smem_buffer[tid*4]); */
        #pragma unroll
        for (int i=0,j=0; i<num_per_thread; i+=4,++j) 
            FetchFloat4(accum[1][i]) = FetchFloat4(block_smem_buffer[j*4*Bx+tid*4]);

        #pragma unroll
        for (int i=0; i<num_per_thread; ++i)
            accum[0][i] += accum[1][i];
    }
    
    __device__ inline void save_to_block_out() {
        #pragma unroll
        for (int i=0,j=0; i<num_per_thread; i+=4,++j)
            FetchFloat4(block_smem_buffer[j*4*Bx+tid*4]) = FetchFloat4(accum[0][i]);
    }
    
    __device__ inline void calculate_gemm() {
        #pragma unroll
        for (int i=0; i<(ldaA/8); ++i) {
            fill_load();
            mma();
            save_buffer();
            __syncthreads();
            load_to_accum();
            move_ab();
        }
        save_to_block_out();
        __syncthreads();
    }
};


const int M = 32;
const int N = 32;
const int K = 32;

const int Bm = 32;
const int Bn = 32;
const int Bx = 128;

__global__ void gemm_tc_kernel(float* a, float*b, float* c){
    float* blockA = a;
    float* blockB = b;
    const int ldaA = K;
    const int ldaB = N;
    
    /* __shared__ float smem_buffer[Bm*Bn]; */
    GemmBlockTile<Bm, Bn, Bx>* gemm = new GemmBlockTile<Bm, Bn, Bx>(blockA, blockB, c, ldaA, ldaB);
    gemm->calculate_gemm();
    delete gemm;

    /* #pragma unroll */
    /* for (int i=0,j=0; i<gemm.num_per_thread; i+=4,++j) */
    /*     FetchFloat4(c[j*gemm.tid*4]) = FetchFloat4(smem_buffer[j*gemm.tid*4]); */
}

int main() {
    float *a = new float[M*K];
    float *b = new float[K*N];
    float *c = new float[M*N];

    /* for (int i=0; i<M*K; ++i) a[i] = (float)rand() / RAND_MAX; */
    /* for (int i=0; i<N*K; ++i) b[i] = (float)rand() / RAND_MAX; */
    for (int i=0; i<M*K; ++i) a[i] = i;
    for (int i=0; i<N*K; ++i) b[i] = i;
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
    
    gemm_tc_kernel<<<1,Bx>>>(da, db, dc);
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
        printf("%d my: %.4f, blas: %.4f ",i,c[i],blas_c[i]);
        printf("delta: %.4f\n",c[i]-blas_c[i]);
        /* } */
    }
    return 0;
}







