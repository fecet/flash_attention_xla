#include <cstdio>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include "utils.h"

namespace py = pybind11;

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

    float *out;

    __device__ LocalSoftmax (
        float *smem_mat, float *smem_reduce, float *gmem_sum, float *gmem_max, float *out
    ): 
    smem_mat(smem_mat), smem_reduce(smem_reduce), gmem_sum(gmem_sum), gmem_max(gmem_max),
    tid(threadIdx.x), warp_id(threadIdx.x/32), lane_id(threadIdx.x%32), loop_id(0),out(out)
    {}

    __device__ inline void load_mat_to_reg() {
        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            thread_mat_reg[i] = smem_mat[
                LdIdx(loop_id*loop_stride+warp_id,32*i+lane_id,Bc)
            ];
        if (loop_id*loop_stride+warp_id == 0){
            printf("row_id:%d, col_id:%d, value:%.2f, smem_mat:%.2f\n",loop_id*loop_stride+warp_id, 32*i+lane_id, thread_mat_reg[i],smem_mat[
                LdIdx(loop_id*loop_stride+warp_id,32*i+lane_id,Bc)
            ]);    
        }
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
            /* if (loop_id == 0 && warp_id == 0) */
                /* printf("tid:%d, i:%.d, v:%.2f, fx:%.2f\n", tid, i, thread_mat_reg[i], thread_fx[i]); */
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
            /* printf("num_loops:%d, loop_id:%d, warp_id:%d, tid:%d, row_id:%d,mi_new:%.2f, in_gmem:%.2f\n",num_loops,loop_id,warp_id,tid,(loop_id*loop_stride+warp_id),mi_new,gmem_max[loop_id*loop_stride+warp_id]); */
        }
        __syncthreads();

        #pragma unroll
        for (int i=0; i<numbers_per_thread; ++i) {
            out[LdIdx(loop_id*loop_stride+warp_id,32*i+lane_id,Bc)] = thread_fx[i] / l_ij;
            if (loop_id*loop_stride+warp_id==0){
            /* printf("%.2f, %.2f\n", thread_fx[i], l_ij); */
            }
        }
        __syncthreads();
    }

    __device__ inline void softmax_loop() {
        loop_id = 0;
        /* #pragma unroll */
        for (int i=0; i<num_loops; ++i,++loop_id) {
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
__global__ void softmaxkernel(float *mat, float *reduce, float *gmem_sum, float *gmem_max, float *out) {
    LocalSoftmax<Br, Bc, Bx>(mat, reduce, gmem_sum, gmem_max,out).softmax_loop();
}

const int Br = 64;
const int Bc = 64;
const int Bx = 512;

using std::cout;

void LocalSoftmaxTest(cudaStream_t stream, void** buffers, const char * opaque, size_t opaque_len) {
    /* printf("Now Using Custom XLA\n"); */
    std::cout << "Now Using Custom XLA" << std::endl;
    float *d_mat = reinterpret_cast<float*>(buffers[0]);
    float *d_out = reinterpret_cast<float*>(buffers[1]);
    /* float *sum_out = reinterpret_cast<float*>(buffers[2]); */
    /* float *max_out = reinterpret_cast<float*>(buffers[3]); */

    float *reduce = new float [Br*32];
    float *gmem_sum = new float [Br];
    float *gmem_max = new float [Br];
    
    for (int i=0; i<Br*32; ++i) reduce[i] = 0.0f;
    for (int i=0; i<Br; ++i) gmem_sum[i] = 0.0f;
    for (int i=0; i<Br; ++i) gmem_max[i] = -INFINITY;
    
    float *d_gmem_max ,*d_reduce, *d_gmem_sum; // *d_gmem_max;
    cudaMalloc(&d_reduce, sizeof(float)*32*Br);
    cudaMalloc(&d_gmem_max, sizeof(float)*Br);
    cudaMalloc(&d_gmem_sum, sizeof(float)*Br);
    
    cudaMemcpy(d_reduce, reduce, sizeof(float)*32*Br, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gmem_sum, gmem_sum, sizeof(float)*Br, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gmem_max, gmem_max, sizeof(float)*Br, cudaMemcpyHostToDevice);
    
    softmaxkernel<Br, Bc, Bx><<<1,Bx,0,stream>>>(d_mat, d_reduce,d_gmem_sum, d_gmem_max, d_out);

    /* void **out = new void * [3]; */
    /* out[0] = (void*)d_out; */
    /* out[1] = (void*)d_gmem_sum; */
    /* out[2] = (void*)d_gmem_max; */
    /* buffers[1] = (void*)out; */
    /* cudaMemcpy(sum_out, gmem_sum, sizeof(float)*Br, cudaMemcpyDeviceToDevice); */
    /* cudaMemcpy(max_out, gmem_sum, sizeof(float)*Br, cudaMemcpyDeviceToDevice); */
    /* cudaMemcpy() */
    /* cudaFree(d_reduce); */
    /* cudaFree(d_gmem_max); */
    /* cudaFree(d_gmem_sum); */
}

PYBIND11_MODULE(flmm, m) {
    m.doc() = "SUSTensorTest of LocalSoftmax";
    m.def("local_softmax", 
        /* & py_cuda_add, */
        [](){
        const char* name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *) &LocalSoftmaxTest, name);
        },
        "Test of LocalSoftmax"
    );
}


/* int main() { */
/*     float *mat = new float [Br*Bc]; */
/*     float *reduce = new float [Br*32]; */
/*     float *gmem_sum = new float [Br]; */
/*     float *gmem_max = new float [Br]; */
/*  */
/*     for (int i=0; i<Br*Bc; ++i) mat[i] = 1.0f; */
/*     for (int i=0; i<Br*32; ++i) reduce[i] = 0.0f; */
/*     for (int i=0; i<Br; ++i) gmem_sum[i] = 0.0f; */
/*     for (int i=0; i<Br; ++i) gmem_max[i] = 0.0f; */
/*  */
/*     float *d_mat, *d_reduce, *d_gmem_sum, *d_gmem_max; */
/*     cudaMalloc(&d_mat, sizeof(float)*Br*Bc); */
/*     cudaMalloc(&d_reduce, sizeof(float)*32*Br); */
/*     cudaMalloc(&d_gmem_max, sizeof(float)*Br); */
/*     cudaMalloc(&d_gmem_sum, sizeof(float)*Br); */
/*  */
/*     cudaMemcpy(d_mat, mat, sizeof(float)*Br*Bc, cudaMemcpyHostToDevice); */
/*     cudaMemcpy(d_reduce, reduce, sizeof(float)*32*Br, cudaMemcpyHostToDevice); */
/*     cudaMemcpy(d_gmem_sum, gmem_sum, sizeof(float)*Br, cudaMemcpyHostToDevice); */
/*     cudaMemcpy(d_gmem_max, gmem_max, sizeof(float)*Br, cudaMemcpyHostToDevice); */
/*      */
/*     softmaxkernel<Br, Bc, Bx><<<1,Bx>>>(d_mat, d_reduce, d_gmem_sum, d_gmem_max); */
/*     cudaMemcpy(mat, d_mat, sizeof(float)*Br*Bc, cudaMemcpyDeviceToHost); */
/*  */
/*     for (int i=0; i<Br*Bc; ++i) { */
/*         printf("%.2f ",mat[i]); */
/*     } */
/*     printf("\n"); */
/*      */
/*     return 0; */
/* } */












