clang++ -std=c++11 --cuda-gpu-arch=sm_80 -shared -fPIC example.cc.cu -o example_end.so -L/opt/cuda/lib64 -lcudart_static -ldl -lrt
