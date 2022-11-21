/* #include <__clang_cuda_builtin_vars.h> */
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <mma.h>
#include <cublas_v2.h>


#define CUDACheck(stat,fn_name) \
    if (stat != cudaSuccess) { \
        printf ("CUDA Failed at %s, with state %d\n", fn_name, stat); \
        return EXIT_FAILURE; \
    } \

// All Row Major
using namespace nvcuda;

#define RowIdx(i,j,width) ((width)*(i)+(j))
#define FetchFloat4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

struct MMAidx {
    int x;
    int y;
};

const int vitual_warp_size = 64;

template <const int Br, const int Bc, const int Bx, const int By>
struct ThreadParam {
    const int tx, ty;
    MMAidx mma_idx;
    int tid;
    int warp_id;
    int nmma_x, nmma_y;
    int num_reg_per_thread;
    
    __device__ explicit ThreadParam(): 
    tx(threadIdx.x), ty(threadIdx.y),
    num_reg_per_thread(Bc*Br/(Bx*By))
    {
        nmma_y = Br / 16;
        nmma_x = Bc / 16;
        tid = ty*Bx + tx;
        warp_id = tid / vitual_warp_size;
        mma_idx.y = warp_id / nmma_x;
        mma_idx.x = warp_id % nmma_x;
    }
};

template <const int Br, const int Bc, const int Bx, const int By>
struct MMAGemm { // For a block
    ThreadParam<Br, Bc, Bx, By> params;
    float* smem_c;
    float* smem_buffer_start;
    float* smem_buffer; // const

    float* a;
    float* b;
    int aw;
    int bw;
    
    float accum[2][Br*Bc/(Bx*By)/4][4] = {0.0f};

    wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    __device__ MMAGemm(
        float* smem_c, float* smem_bfs, 
        float* a_start, float* b_start,
        int aw, int bw
    ):
        params(),
        smem_c(smem_c), smem_buffer_start(smem_bfs),
        aw(aw), bw(bw)
    {
        a = a_start + 16*aw*params.mma_idx.y;
        b = b_start + 16*params.mma_idx.x;
        /* smem_buffer = smem_bfs + (16*16)*(params.mma_idx.y*params.nmma_x+params.mma_idx.x); */
        smem_buffer = smem_bfs + 16*bw*params.mma_idx.y + 16*params.mma_idx.x;
        /* printf("tid:%d warpid:%d mma_idx=%d,%d\n", params.tid, params.warp_id,params.mma_idx.y,params.mma_idx.x);    */
    }

    __device__ void fill_load() {
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, a, aw);
        wmma::load_matrix_sync(b_frag, b, bw);
    }

    __device__ inline void mma() {wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);}
    __device__ inline void store() {wmma::store_matrix_sync(smem_buffer, c_frag, bw, wmma::mem_row_major);}
    __device__ inline void move_ab() { a += 8; b += 8*bw;}
    __device__ inline void load_to_accum() {
        // smem_buffer is 16x16
        // Assume that Br*Bc/(Bx*By) = 4 or 8 or ...
        int stride = By*Bx*4;
        int tid = params.tid;
        #pragma unroll
        for(int i=0, j=0; i<Br*Bc; i+=stride, ++j) 
            FetchFloat4(accum[1][j][0]) = FetchFloat4(smem_buffer_start[tid*4]);

        #pragma unroll
        for (int y=0; y<Br*Bc/(Bx*By)/4; ++y) {
            #pragma unroll
            for (int x=0; x<4; ++x)
                accum[0][y][x] += accum[1][y][x];
        }
    }
    __device__ inline void copy_to_smem() {
        int stride = By*Bx*4;
        int tid = params.tid;

        #pragma unroll
        for(int i=0, j=0; i<Br*Bc; i+=stride, ++j) 
            /* FetchFloat4(accum[1][j][0]) = FetchFloat4(smem_buffer_start[tid*4]); */
            FetchFloat4(smem_c[tid*4]) = FetchFloat4(accum[0][j][0]);
    }
    
    __device__ inline void calculate_mma_all() {
        #pragma unroll
        for (int i=0; i<(aw/8); ++i) {
            fill_load();
            mma();
            store();
            __syncthreads();
            load_to_accum();
            move_ab();
        }
        copy_to_smem();
        __syncthreads();
    }
};

const int M = 32;
const int N = 32;
const int K = 32;

const int Br = 32;
const int Bc = 32;
const int Bx = 16;
const int By = 16;

__global__ void gemm_tc_kernel(float* a, float* b, float* c) {

    const int nmma_y = Br / 16;
    const int nmma_x = Bc / 16;
    const int n_floats = nmma_y * nmma_x * 16 * 16;
    __shared__ float smem_c[n_floats];
    __shared__ float smem_buffer[n_floats];
    MMAGemm<Br, Bc, Bx, By> gemm(smem_c, smem_buffer, a, b, 32, 32);
    gemm.calculate_mma_all();
    for (int i=0; i<n_floats; ++i) {
        c[i] = smem_c[i];
    }
}

int main() {
    float *a = new float[32*32];
    float *b = new float[32*32];
    float *c = new float[32*32];

    for (int i=0; i<32*32; ++i) {
        a[i] = i;
        b[i] = i;
        c[i] = 0.0f;
    }


    float *da, *db;
    float *dc;
    size_t nbytes = sizeof(float) * 1024;
    size_t nbytes_float = sizeof(float) * 1024;
    cudaError_t stat;
    stat = cudaMalloc(&da, nbytes);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&db, nbytes);
    CUDACheck(stat, "malloc");
    stat = cudaMalloc(&dc, nbytes_float);
    CUDACheck(stat, "malloc");

    stat = cudaMemcpy(da, a, nbytes, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(db, b, nbytes, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");
    stat = cudaMemcpy(dc, c, nbytes_float, cudaMemcpyHostToDevice);
    CUDACheck(stat, "memcpy");

    dim3 block_dim;
    block_dim.x = Bx;
    block_dim.y = By;
    gemm_tc_kernel<<<1,block_dim>>>(da, db, dc);
    stat = cudaGetLastError();
    CUDACheck(stat, "kernel");
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, nbytes_float, cudaMemcpyDeviceToHost);

    float *blas_c = new float [1024];
    for (int i=0; i<1024; ++i) blas_c[i] = 0.0f;
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    cudaMemcpy( dc, blas_c, sizeof(float)*1024, cudaMemcpyHostToDevice);
    cublasSgemm_v2(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, db, N, da, K, &beta, dc, N);
    cudaMemcpy(blas_c, dc, sizeof(float)*1024, cudaMemcpyDeviceToHost);

    for (int i=0; i<32; ++i)
        for (int j=0; j<32; ++j) {
            float c_ij = c[i*32+j];
            float c1_ij = blas_c[i*32+j];
            printf("[%d][%d]: delta = %.2f\n",i,j,c_ij-c1_ij);
        }

    return 0;
}


