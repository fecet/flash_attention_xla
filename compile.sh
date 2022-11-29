clang++ -std=c++11 --cuda-gpu-arch=sm_80 -shared -fPIC fmha_Nd.cu -o fmha.so -L/opt/cuda/lib64 -lcudart_static -ldl -lrt
