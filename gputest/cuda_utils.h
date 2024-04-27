/**
 * Header File for Utility Functions for the CUDA solver
 * This header file provides wrappers for efficient reduction and argmax Kernels
*/

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <algorithm>
#include <math.h>


#define BLOCKDIM 1024

/**
 * CPU wrapper for reduce sum implementation. The scratch inputs are optional
 * @param device_in The array to be summed over, located in device memory
 * @param len The length of the array
 * @param device_result_out A pointer to a single float where the summation
 *                          result will be stored
 * @param device_block_scratch A pointer to the scratch array in device memory.
 *                             size == ceil(len / 2048).
 * @remark This function will allocate its own block scratch if not provided.
*/
void cuda_reduce_sum(float *device_in, size_t len, float *device_result_out,
                     float *device_block_scratch = NULL);

/**
 * CPU wrapper for reduce sum implementation. This function will also count
 * the number of non-zeros in the float array being reduced
 * @param device_in The array to be summed over, located in device memory
 * @param len The length of the array
 * @param device_sum_out A pointer to a single float where the summation
 *                          result will be stored
 * @param device_count_out A pointer to a single int where the count result
 *                          will be stored.
 * @remark This function will allocate its own block scratch if not provided.
*/
void cuda_reduce_sum_count_nonzeros(float *device_in, size_t len,
                float *device_sum_out, int *device_count_out,
                float *device_block_scratch_sum = NULL,
                int *device_block_scratch_count = NULL);

/**
 * CPU wrapper for the max / argmax implementation.
 * @param device_in_num The element array residing in device memory
 * @param device_in_idx The corresponding indecies of the element array.
 *                      use NULL to use the vanilla index for device_in.
 * @param len The length of the array
 * @param device_max_out A pointer to a single float for the maximum value
 * @param device_argmax_out A pointer to a single int for the index of max
 * @remark This function will allocate its own block scratch if not provided.
*/
void cuda_max(float *device_in_num, int *device_in_idx, size_t len,
    float *device_max_out, int *device_argmax_out, 
    float *device_scratch_num = NULL, 
    int *device_scratch_idx = NULL);

/**
 * CPU wrapper for argmax implementation. 
*/
void cuda_argmax(float *device_in, size_t len,
                 int *device_argmax_out, 
                 float* device_block_scratch = NULL,
                 int *device_block_scratch_index = NULL);


#endif