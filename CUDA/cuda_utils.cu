#include <math.h>
#include <stdio.h>


#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_utils.h"



int CPU_ceil_xdivy(int X, int Y){
    return (X + (Y - 1)) / Y;
}

/**********************
 * CUDA Kernels
***********************/

/**
 * The following implementations are inspired by the following resources:
 * https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
 * https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
*/

__global__ void kernel_reduce_sum(float *device_in, int len, float *device_block_out){
    __shared__ float temp[BLOCKDIM * 2];
    int chunksize = blockDim.x * 2; // The work region current block is responsible for
    int start_idx = blockIdx.x * chunksize;
    int tid = threadIdx.x;
    // Copy global data into the cache to avoid destructive modification
    if(start_idx + tid < len){
        temp[tid] = device_in[start_idx + tid];
    }
    else{
        temp[tid] = 0.0f; // Pad with 0's if out of bounds
    }
    if(start_idx + tid + blockDim.x < len){
        temp[tid + blockDim.x] = device_in[start_idx + tid + blockDim.x];
    }
    else{
        temp[tid + blockDim.x] = 0.0f;
    }
    __syncthreads();
    // Now, perform reduction using sequential addressing
    int stride = blockDim.x;
    while(stride > 0){
        if(tid < stride){
            temp[tid] += temp[tid + stride];
        }
        stride >>= 1; // equivalent to stride /= 2.
        __syncthreads();
    }
    // temp[0] holds the reduced result. Write that to the block scratch memory
    if(tid == 0){
        device_block_out[blockIdx.x] = temp[0];
    }
}

/**
 * Kernel for cuda_reduce_sum_count_nonzeros. Will only be called once
*/
__global__ void kernel_reduce_sum_count(float *device_in, int len, 
    float *device_sum_out, int *device_count_out){
    // Utilizes 16 Kib of shared cache
    __shared__ float temp_sum[BLOCKDIM * 2];
    __shared__ int temp_count[BLOCKDIM * 2];

    int chunksize = blockDim.x * 2; // The work region current block is responsible for
    int start_idx = blockIdx.x * chunksize;
    int tid = threadIdx.x;
    // Copy global data into the cache, generate an indicator array of non-zero
    float read;
    if(start_idx + tid < len){
        read = device_in[start_idx + tid];
        temp_sum[tid] = read;
        temp_count[tid] = (fabsf(read) <= 1e-12) ? 0 : 1; 
    }
    else{
        temp_sum[tid] = 0.0f; // Pad with 0's if out of bounds
        temp_count[tid] = 0;
    }
    if(start_idx + tid + blockDim.x < len){
        read = device_in[start_idx + tid + blockDim.x];
        temp_sum[tid + blockDim.x] = read;
        temp_count[tid + blockDim.x] = (fabsf(read) <= 1e-12) ? 0 : 1; 
    }
    else{
        temp_sum[tid + blockDim.x] = 0.0f;
        temp_count[tid + blockDim.x] = 0;
    }
    __syncthreads();
    // Perform float reduction sum on temp_sum
    // Perform integer reduction sum on temp_count
    int stride = blockDim.x;
    while(stride > 0){
        if(tid < stride){
            temp_sum[tid] += temp_sum[tid + stride];
            temp_count[tid] += temp_count[tid + stride];
        }
        stride >>= 1; // equivalent to stride /= 2.
        __syncthreads();
    }
    // temp[0] holds the reduced result. Write that to the block scratch memory
    if(tid == 0){
        device_sum_out[blockIdx.x] = temp_sum[0];
        device_count_out[blockIdx.x] = temp_count[0];
    }
}


/**
 * An integer version of the kernel reduce sum. Not accessible via the interface.
*/
__global__ void kernel_reduce_sum_int(int *device_in, int len, int *device_block_out){
    __shared__ float temp[BLOCKDIM * 2];
    int chunksize = blockDim.x * 2; // The work region current block is responsible for
    int start_idx = blockIdx.x * chunksize;
    int tid = threadIdx.x;
    // Copy global data into the cache to avoid destructive modification
    if(start_idx + tid < len){
        temp[tid] = device_in[start_idx + tid];
    }
    else{
        temp[tid] = 0; // Pad with 0's if out of bounds
    }
    if(start_idx + tid + blockDim.x < len){
        temp[tid + blockDim.x] = device_in[start_idx + tid + blockDim.x];
    }
    else{
        temp[tid + blockDim.x] = 0;
    }
    __syncthreads();
    // Now, perform reduction using sequential addressing
    int stride = blockDim.x;
    while(stride > 0){
        if(tid < stride){
            temp[tid] += temp[tid + stride];
        }
        stride >>= 1; // equivalent to stride /= 2.
        __syncthreads();
    }
    // temp[0] holds the reduced result. Write that to the block scratch memory
    if(tid == 0){
        device_block_out[blockIdx.x] = temp[0];
    }
}


__global__ void kernel_max(float *device_in_num, int *device_in_idx, int len,
    float *device_max_out, int *device_argmax_out){
    __shared__ float temp_num[BLOCKDIM * 2];
    __shared__ int temp_idx[BLOCKDIM * 2];
    int chunksize = blockDim.x * 2; // The work region current block is responsible for
    int start_idx = blockIdx.x * chunksize;
    int tid = threadIdx.x;
    // The indecies of the input numbers are not specified.
    // Use the default corresponding indecies
    if(device_in_idx == NULL){
        if(start_idx + tid < len){
            temp_num[tid] = device_in_num[start_idx + tid];
            temp_idx[tid] = start_idx + tid;
        }
        else{
            temp_num[tid] = -1e10; // Use some very large negative numbers
            temp_idx[tid] = start_idx + tid;
        }
        if(start_idx + tid + blockDim.x < len){
            temp_num[tid + blockDim.x] = device_in_num[start_idx + tid + blockDim.x];
            temp_idx[tid + blockDim.x] = start_idx + tid + blockDim.x;
        }
        else{
            temp_num[tid + blockDim.x] = -1e10;
            temp_idx[tid + blockDim.x] = start_idx + tid + blockDim.x;
        }
    }
    else{
        if(start_idx + tid < len){
            temp_num[tid] = device_in_num[start_idx + tid];
            temp_idx[tid] = device_in_idx[start_idx + tid];
        }
        else{
            temp_num[tid] = -1e10; // Use some very large negative numbers
            temp_idx[tid] = 0;
        }
        if(start_idx + tid + blockDim.x < len){
            temp_num[tid + blockDim.x] = 
                device_in_num[start_idx + tid + blockDim.x];
            temp_idx[tid + blockDim.x] = 
                device_in_idx[start_idx + tid + blockDim.x];
        }
        else{
            temp_num[tid + blockDim.x] = -1e10;
            temp_idx[tid + blockDim.x] = 0;
        }
    }
    __syncthreads(); // The lengthy copying phase is finally over!

    // Now, perform reduction using sequential addressing
    int stride = blockDim.x;
    while(stride > 0){
        if(tid < stride){
            if(temp_num[tid] < temp_num[tid + stride]){
                temp_num[tid] = temp_num[tid + stride];
                temp_idx[tid] = temp_idx[tid + stride];
            }
        }
        stride >>= 1; // equivalent to stride /= 2.
        __syncthreads();
    }
    // temp[0] holds the reduced result. Write that to the block scratch memory
    if(tid == 0){
        device_max_out[blockIdx.x] = temp_num[0];
        device_argmax_out[blockIdx.x] = temp_idx[0];
    }
}


/**********************
 * CPU Functions
***********************/

/**
 * An integer version of the cuda_reduce sum routine. But this function
 * is not provided in the interface.
*/
void cuda_reduce_sum_int(int *device_in, size_t len, int *device_result_out){
    // Each block will be responsible for 2048 elements in the input
    int num_blocks = CPU_ceil_xdivy(static_cast<int>(len), BLOCKDIM * 2);

    int * device_block_scratch;
    cudaMalloc(&device_block_scratch, sizeof(int) * num_blocks);
    cudaMemset(device_block_scratch, 0, sizeof(int) * num_blocks);

    // Perform Multi-block reduction
    kernel_reduce_sum_int<<<num_blocks, BLOCKDIM>>>(device_in, static_cast<int>(len), device_block_scratch);

    // Now aggregate the block outputs. One final block should do for less than 2048 blocks.
    // Scratch serves as the input, directly write the result to device_result out.
    if(num_blocks <= BLOCKDIM * 2){
        kernel_reduce_sum_int<<<1, BLOCKDIM>>>(device_block_scratch, num_blocks, device_result_out);
    }
    else{ 
        // One block is not sufficient, call this function recursively
        cuda_reduce_sum_int(device_block_scratch, num_blocks, device_result_out);
    }
    cudaFree(device_block_scratch);
}



void cuda_reduce_sum(float *device_in, size_t len, float *device_result_out,
                     float *device_block_scratch){
    // Each block will be responsible for 2048 elements in the input
    int num_blocks = CPU_ceil_xdivy(static_cast<int>(len), BLOCKDIM * 2);
    bool alloc_scratch = false;

    if(device_block_scratch == NULL){
        // Caller did not provide a scratch space, allocate new.
        cudaMalloc(&device_block_scratch, sizeof(float) * num_blocks);
        alloc_scratch = true;
    }

    // Initialize the scratch space to 0 before use
    cudaMemset(device_block_scratch, 0, sizeof(float) * num_blocks);

    // Perform Multi-block reduction
    kernel_reduce_sum<<<num_blocks, BLOCKDIM>>>(device_in, static_cast<int>(len), device_block_scratch);

    // Now aggregate the block outputs. One final block should do for less than 2048 blocks.
    // Scratch serves as the input, directly write the result to device_result out.
    if(num_blocks <= BLOCKDIM * 2){
        kernel_reduce_sum<<<1, BLOCKDIM>>>(device_block_scratch, num_blocks, device_result_out);
    }
    else{ 
        // One block is not sufficient, call this function recursively and
        // do not supply a scratch.
        cuda_reduce_sum(device_block_scratch, num_blocks, device_result_out, NULL);
    }

    // De-allocate the scratch space if it is allocated by this function.
    if(alloc_scratch){
        cudaFree(device_block_scratch);
    }
}

void cuda_reduce_sum_count_nonzeros(float *device_in, size_t len,
                float *device_sum_out, int *device_count_out,
                float *device_block_scratch_sum,
                int *device_block_scratch_count){
    // Each block will be responsible for 2048 elements in the input
    int num_blocks = CPU_ceil_xdivy(static_cast<int>(len), BLOCKDIM * 2);
    bool alloc_scratch = false;

    // This function will allocate its own space if either scratch space is NULL
    if(device_block_scratch_count == NULL || device_block_scratch_sum == NULL){
        cudaMalloc(&device_block_scratch_sum, sizeof(float) * num_blocks);
        cudaMalloc(&device_block_scratch_count, sizeof(int) * num_blocks);
        alloc_scratch = true;
    }
    // Initialize the scratch space to 0 before use
    cudaMemset(device_block_scratch_sum, 0, sizeof(float) * num_blocks);
    cudaMemset(device_block_scratch_count, 0, sizeof(int) * num_blocks);

    // Launch the dedicated kernel for reduce_sum_count
    kernel_reduce_sum_count<<<num_blocks, BLOCKDIM>>>(device_in, 
        static_cast<int>(len), device_block_scratch_sum, device_block_scratch_count);

    // For the remaining steps, perform float reduction on scratch sum
    // perform integer reduction on scratch count.
    if(num_blocks <= BLOCKDIM * 2){
        kernel_reduce_sum<<<1, BLOCKDIM>>>(device_block_scratch_sum, num_blocks, device_sum_out);
        kernel_reduce_sum_int<<<1, BLOCKDIM>>>(device_block_scratch_count, num_blocks, device_count_out);
    }
    else{ 
        cuda_reduce_sum(device_block_scratch_sum, num_blocks, device_sum_out, NULL);
        cuda_reduce_sum_int(device_block_scratch_count, num_blocks, device_count_out);
    }

    // Free the memory this function is responsible for
    if(alloc_scratch){
        cudaFree(device_block_scratch_sum);
        cudaFree(device_block_scratch_count);
    }
}


/**
 * This is an augmented function for argmax. It returns both the
 * max value and the argmax index.
 * 
*/
void cuda_max(float *device_in_num, int *device_in_idx, size_t len,
    float *device_max_out, int *device_argmax_out, 
    float *device_scratch_num, int *device_scratch_idx){
    int num_blocks = CPU_ceil_xdivy(static_cast<int>(len), BLOCKDIM * 2);
    bool alloc_scratch = false;

    if(device_scratch_num == NULL || device_scratch_idx == NULL){
        // Caller did not provide a scratch space, allocate new.
        cudaMalloc(&device_scratch_num, sizeof(float) * num_blocks);
        cudaMalloc(&device_scratch_idx, sizeof(int) * num_blocks);
        alloc_scratch = true;
    }

    // Initialize the scratch space to 0 before use
    cudaMemset(device_scratch_num, 0, sizeof(float) * num_blocks);
    cudaMemset(device_scratch_idx, 0, sizeof(int) * num_blocks);

    // Launch the max kernel
    kernel_max<<<num_blocks, BLOCKDIM>>>(device_in_num, device_in_idx, len, 
                    device_scratch_num, device_scratch_idx);

    // One more CUDA block would be sufficient
    if(num_blocks <= BLOCKDIM * 2){
       kernel_max<<<1, BLOCKDIM>>>(device_scratch_num, device_scratch_idx, 
                    num_blocks, device_max_out, device_argmax_out);
    }
    else{ // Recursively call the current function, but do not supply scratch
        cuda_max(device_scratch_num, device_scratch_idx, num_blocks,
             device_max_out, device_argmax_out, NULL, NULL);
    }    
    // De-allocate the scratch space if it is allocated by this function.
    if(alloc_scratch){
        cudaFree(device_scratch_num);
        cudaFree(device_scratch_idx);
    }
}


void cuda_argmax(float *device_in, size_t len,
                 int *device_argmax_out, 
                 float* device_block_scratch,
                 int *device_block_scratch_index){
    // Allocate a single cuda item to store the max value.
    // It will be discarded later
    float *tmp;
    cudaMalloc(&tmp, sizeof(float));

    // use cuda max as the subroutine:
    cuda_max(device_in, NULL, len, tmp, device_argmax_out, device_block_scratch, device_block_scratch_index);

    // We do not need the max numerical value.
    cudaFree(tmp);
}