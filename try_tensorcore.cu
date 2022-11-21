/* #include <__clang_cuda_builtin_vars.h> */
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

#define CUDACheck(stat,fn_name) \
    if (stat != cudaSuccess) { \
        printf ("CUDA Failed at %s, with state %d\n", fn_name, stat); \
        return EXIT_FAILURE; \
    } \

#define RowIdx(i,j,width) ((width)*(i)+(j))

const int M = 16;
const int N = 16;
const int K = 16;

struct MMAGemm { // For a block
    volatile float* smem_c;
    const float* a;
    const unsigned int wa;
    const float* b;
    const int wb;

    wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

    __device__ explicit MMAGemm(volatile float* s, const float* a, const float* b, const int wa, const int wb):smem_c(s), a(a), b(b), wa(wa), wb(wb) {
        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, a, wa);
        wmma::load_matrix_sync(b_frag, b, wb);
    }

};

__device__ void load_smem(float* a, float* b, volatile float* smem_a, volatile float* smem_b) {
    smem_a[threadIdx.y*blockDim.x+threadIdx.x] = a[RowIdx(threadIdx.y, threadIdx.x, K)];
    smem_b[threadIdx.y*blockDim.x+threadIdx.x] = b[RowIdx(threadIdx.y, threadIdx.x, K)];
    /* smem_a[threadIdx.y][threadIdx.x] = a[RowIdx(threadIdx.y, threadIdx.x, K)]; */
    /* smem_b[threadIdx.y][threadIdx.x] = b[RowIdx(threadIdx.y, threadIdx.x, N)]; */
}

__global__ void wmma_kernel(float *a, float *b, float *c) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockDim.x, by = blockDim.y;
    __shared__ float smem_a[M*K];
    __shared__ float smem_b[K*N];
    __shared__ float smem_c[M*N];
    load_smem(a, b, smem_a, smem_b);
    __syncthreads();

    MMAGemm(smem_c, a, b, 16, 16);

    wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, smem_a, 16);
    wmma::load_matrix_sync(b_frag, smem_b, 16);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    /* wmma::store_matrix_sync(smem_c,c_frag,16,wmma::mem_row_major); */
    wmma::store_matrix_sync(c,c_frag,16,wmma::mem_row_major);
    printf("tx:%d, ty:%d, smem_c:%.2f\n",tx,ty,(float)smem_c[ty*bx+tx]);
}

void print_matrix(float* mat, int height, int width) {
    for (int i=0; i<height; ++i) {
        for (int j=0; j<width; ++j) {
            int idx = RowIdx(i, j, width);
            printf("%.2f ",mat[idx]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    float *a = new float[256];
    float *b = new float[256];
    float *c = new float[256];

    for (int i=0; i<256; ++i) {
        a[i] = 1.0f;
        b[i] = 1.0f;
        c[i] = 1.0f;
    }


    float *da, *db;
    float *dc;
    size_t nbytes = sizeof(float) * 256;
    size_t nbytes_float = sizeof(float) * 256;
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

    dim3 block_dim (16,16);
    wmma_kernel<<<1,block_dim>>>(da, db, dc);
    stat = cudaGetLastError();
    CUDACheck(stat, "kernel");
    cudaDeviceSynchronize();

    cudaMemcpy(c, dc, nbytes_float, cudaMemcpyDeviceToHost);
    print_matrix(c, 16, 16);

    return 0;
}

