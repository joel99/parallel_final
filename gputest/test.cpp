/**
 * Testing file for different implementations of scatter reduce
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <cmath>

#include "cuda_utils.h"
#include "aux.h"



// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()

void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << "-n <test dimension>\n";
    return;
}

bool is_zero(float x){
    return std::fabs(x) <= 1e-12;
}

int ceil_xdivy(int X, int Y){
    return (X + (Y - 1)) / Y;
}


float f_rand(float low, float high){
    return static_cast <float> (rand()) /
        (static_cast <float> (RAND_MAX/(high-low)));
}

// Random integer [low, high)
int i_rand(int low, int high){
    return rand()%(high - low) + low; 
}


/************************
 * CPU Implementations
************************/

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

float sum(std::vector<float> vec){
    float out = 0.0f;
    for(float &elt: vec){
        out += elt;
    }
    return out;
}

void sum_count_nonzero(std::vector<float> vec, float &sum_out, int &count_out){
    sum_out = 0.0f;
    count_out = 0;
    for(float &elt : vec){
        sum_out += elt;
        if(!is_zero(elt)) count_out += 1;
    }
    return;
}


/**
 * Testing Routine
*/
int main(int argc, char **argv) {
    // Initialization Stage
    long input_dim = 0L;
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "n:")) != -1) {
        switch (opt) {
        case 'n':
            input_dim = atol(optarg);
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }
    if(input_dim <= 0){
        usage(argv[0]);
        exit(1);
    }

    // Allocate space both on CPU and GPU
    std::vector<float> data_in(input_dim);
    float *gpu_data = device_alloc_float(input_dim);
    

    // Allocate temporary space for the GPU to store the results
    float *gpu_sum_out  = device_alloc_float(1);
    int *gpu_count_out  = device_alloc_int(1);


    // Populate the CPU vector
    for (long i = 0; i < input_dim; i++){
        data_in[i] = f_rand(1e-4, 1.0f);
    }

    copy_to_device_float(&(data_in[0]), gpu_data, input_dim);

    std::cout << "Testing Reduction Sum...\n";

    auto cpu_start = timestamp;
    float cpu_sum = sum(data_in);
    auto cpu_end = timestamp;

    auto gpu_start = timestamp;
    cuda_reduce_sum(gpu_data, input_dim, gpu_sum_out);
    auto gpu_end = timestamp;
    
    float gpu_sum;
    copy_from_device_float(&gpu_sum, gpu_sum_out, 1);

    std::cout << "CPU computation time:   " << TIME(cpu_start, cpu_end) << "\n";
    std::cout << "GPU computation time: " << TIME(gpu_start, gpu_end) << "\n";
    std::cout << "Speedup: " << TIME(cpu_start, cpu_end) / TIME(gpu_start, gpu_end) << "\n";
    std::cout << "CPU result: " << cpu_sum << " |  GPU result: " << gpu_sum << " | Diff: " << std::fabs(cpu_sum - gpu_sum) << "\n";

    // Now modify the data array to contain some zeros
    int tmp; // 80% zeros now
    for (long i = 0; i < input_dim; i++){
        tmp = i_rand(0, 5);
        if(tmp != 0){
            data_in[i] = 0.0;
        } 
    }

    copy_to_device_float(&(data_in[0]), gpu_data, input_dim);

    std::cout << "Testing Reduction Sum and Count Non-zeros...\n";
    int cpu_count;
    int gpu_count;

    cpu_start = timestamp;
    sum_count_nonzero(data_in, cpu_sum, cpu_count);
    cpu_end = timestamp;

    gpu_start = timestamp;
    cuda_reduce_sum_count_nonzeros(gpu_data, input_dim, gpu_sum_out, gpu_count_out);
    gpu_end = timestamp;

    copy_from_device_float(&gpu_sum, gpu_sum_out, 1);
    copy_from_device_int(&gpu_count, gpu_count_out, 1);

    std::cout << "CPU computation time:   " << TIME(cpu_start, cpu_end) << "\n";
    std::cout << "GPU computation time: " << TIME(gpu_start, gpu_end) << "\n";
    std::cout << "Speedup: " << TIME(cpu_start, cpu_end) / TIME(gpu_start, gpu_end) << "\n";
    std::cout << "CPU result: " << cpu_sum << " |  GPU result: " << gpu_sum << " | Diff: " << std::fabs(cpu_sum - gpu_sum) << "\n";
    std::cout << "CPU count: " << cpu_count << " | GPU count: " << gpu_count << "\n";

    std::cout << "Testing Max and Argmax...\n";
    int cpu_argmax, gpu_argmax;
    float cpu_max, gpu_max;
    // Simply repurpose the sum and count pointers as max and argmax pointers;
    cpu_start = timestamp;
    cpu_argmax = arg_max(data_in);
    cpu_max = data_in[cpu_argmax];
    cpu_end = timestamp;

    gpu_start = timestamp;
    cuda_max(gpu_data, NULL, input_dim, gpu_sum_out, gpu_count_out);
    gpu_end = timestamp;

    copy_from_device_float(&gpu_max, gpu_sum_out, 1);
    copy_from_device_int(&gpu_argmax, gpu_count_out, 1);

        std::cout << "CPU computation time:   " << TIME(cpu_start, cpu_end) << "\n";
    std::cout << "GPU computation time: " << TIME(gpu_start, gpu_end) << "\n";
    std::cout << "Speedup: " << TIME(cpu_start, cpu_end) / TIME(gpu_start, gpu_end) << "\n";
    std::cout << "CPU result: (" << cpu_max << " @ " << cpu_argmax << ") |  GPU result: ("\
         << gpu_max << " @ " << gpu_argmax << ")\n";

    device_free(gpu_sum_out);
    device_free(gpu_count_out);
    device_free(gpu_data);

    std::cout << "Benchmark End\n";
}