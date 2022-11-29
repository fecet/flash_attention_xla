#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__
void py_cuda_add_kernel(const float* a, const float* b, float* out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    out[idx] = a[idx] + b[idx];
}

void py_cuda_add(cudaStream_t stream, void** buffers, const char * opaque, size_t opaque_len) {
    std::cout << "Now use custom XLA!" << std::endl;
    /* std::cout << opaque << std::endl; */
    const float* a = reinterpret_cast<float*>(buffers[0]);
    const float* b = reinterpret_cast<float*>(buffers[1]);
    float* c = reinterpret_cast<float*>(buffers[2]);
    py_cuda_add_kernel<<<8,16,0,stream>>>(a, b, c);
}

PYBIND11_MODULE(example, m) {
    m.doc() = "haha,pyadd";
    m.def("py_add", 
        /* & py_cuda_add, */
        [](){
        const char* name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *) &py_cuda_add, name);
        },
        "HAHA! PY ADDED!"
    );
}

