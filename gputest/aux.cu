#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "aux.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int *device_alloc_int(int size){
    int *out;
    gpuErrchk(cudaMalloc(&out, sizeof(int) * size));
    return out;
}

float *device_alloc_float(int size){
    float *out;
    gpuErrchk(cudaMalloc(&out, sizeof(float) * size));
    return out;
}

void copy_to_device_int(int *host_ptr, int *device_ptr, int count){
    gpuErrchk(cudaMemcpy(device_ptr, host_ptr, sizeof(int) * count, cudaMemcpyHostToDevice));
}

void copy_from_device_int(int *host_ptr, int *device_ptr, int count){
    gpuErrchk(cudaMemcpy(host_ptr, device_ptr, sizeof(int) * count, cudaMemcpyDeviceToHost));
}

void copy_to_device_float(float *host_ptr, float *device_ptr, int count){
    gpuErrchk(cudaMemcpy(device_ptr, host_ptr, sizeof(float) * count, cudaMemcpyHostToDevice));
}

void copy_from_device_float(float *host_ptr, float *device_ptr, int count){
    gpuErrchk(cudaMemcpy(host_ptr, device_ptr, sizeof(float) * count, cudaMemcpyDeviceToHost));
}

void device_free(void *ptr){
    gpuErrchk(cudaFree(ptr));
}