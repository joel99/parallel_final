#ifndef AUX_H
#define AUX_H

/**
 * Some auxillary functions for CUDA Testing
*/

int *device_alloc_int(int size);

float *device_alloc_float(int size);

void copy_to_device_int(int *host_ptr, int *device_ptr, int count);

void copy_from_device_int(int *host_ptr, int *device_ptr, int count);

void copy_to_device_float(float *host_ptr, float *device_ptr, int count);

void copy_from_device_float(float *host_ptr, float *device_ptr, int count);

void device_free(void *ptr);


#endif