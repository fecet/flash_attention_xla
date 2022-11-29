#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <stdio.h>
#include <cublas_v2.h>
#include <iostream>
#include <sys/time.h>

template<typename T>
struct matrix_t{
    int height;
    int width;
    T *elem;
};

template<typename T>
struct vector_t{
    unsigned long int length;
    T *elem;
};

#define ColIdx(mat,i,j) ((mat.height)*(j)+(i))
#define RowIdx(mat,i,j) ((mat.width)*(i)+(j))
#define NBytes(mat, type) (sizeof(type)*mat.height*mat.width) 
#define FetchFloat4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FetchFloat2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
#define LdIdx(row, col, ld) ((row) * (ld) + (col))

template<typename T>
inline void print_matrix(matrix_t<T> mat, int type) {
    int idx;
    for (int i=0; i<mat.height; i++) {
        for (int j=0; j<mat.width; j++) {
            idx = type==0 ? ColIdx(mat, i, j) : RowIdx(mat, i, j);
            std::cout << mat.elem[idx] << ' ';
        }
        printf("\n");
    }
    printf("\n");
}

#define CUBLASCheck(stat,fn_name) \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        printf ("CUBLAS failed at %s, with state %d\n", fn_name, stat); \
        return EXIT_FAILURE; \
    } \

#define CUDACheck(stat,fn_name) \
    if (stat != cudaSuccess) { \
        printf ("CUDA Failed at %s, with state %d\n", fn_name, stat); \
        printf ("Error Message: %s\n", cudaGetErrorString(stat)); \
        return EXIT_FAILURE; \
    } \

#define Timer(seg_name,seg) do { \
    struct timeval tval_before, tval_after, tval_result; \
    gettimeofday(&tval_before, NULL); \
    seg \
    gettimeofday(&tval_after, NULL); \
    timersub(&tval_after, &tval_before, &tval_result); \
    long int seconds = (long int)tval_result.tv_sec; \
    long int u_seconds = (long int)tval_result.tv_usec; \
    long double m_seconds = (long double) seconds * 1000 + (long double)u_seconds / 1000.0; \
    printf("%s time elapsed: %.3Lf ms\n",seg_name,m_seconds); \
    } while(0) \

#define TimerInit \
    struct timeval tval_before, tval_after, tval_result; \
    long int seconds; long int u_seconds; long double m_seconds; \

#define TimerStart \
    gettimeofday(&tval_before, NULL); \

#define TimerEnd(seg_name) \
    gettimeofday(&tval_after, NULL); \
    timersub(&tval_after, &tval_before, &tval_result); \
    seconds = (long int)tval_result.tv_sec; \
    u_seconds = (long int)tval_result.tv_usec; \
    m_seconds = (long double) seconds * 1000 + (long double)u_seconds / 1000.0; \
    printf("%s time elapsed: %.3Lf ms\n",seg_name,m_seconds); \

#endif
