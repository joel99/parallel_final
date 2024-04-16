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

#include <omp.h>

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
#define TASKSIZE 32
#define SIMDWIDTH 32
#define PRECISION 1e-4

void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << "-i <input dimension> -o <output dimension> -n <thread count> -m <mode> [-l <number of locks>]\n";
    std::cout << "-m: 'L': lock based implementation, 'R': reduction based implementation\n";
    return;
}

bool is_zero(double x){
    return std::fabs(x) <= PRECISION;
}

int ceil_xdivy(int X, int Y){
    return (X + (Y - 1)) / Y;
}


double f_rand(double low, double high){
    return static_cast <double> (rand()) /
        (static_cast <double> (RAND_MAX/(high-low)));
}

// Random integer [low, high)
int i_rand(int low, int high){
    return rand()%(high - low) + low; 
}


/********************************************************************
 * Scatter Reduce Implementations
*********************************************************************/

void scatter_reduce(std::vector<int> &index, std::vector<double> &in,
    std::vector<double> &out){
    size_t n = index.size();
    int j;
    for(size_t i = 0; i < n; i++){
        j = index[i];
        out[j] += in[i];
    }
}

/**
 * A simple lock based scatter reduce without any data preprocessing
 * @param locks a vector of initialized omp locks
*/
void lock_scatter_reduce(std::vector<double> &data_in,
                         std::vector<int> &data_index,
                         std::vector<double> &data_out,
                         std::vector<omp_lock_t> &locks){
    // Compute the span of each lock
    int n = static_cast<int>(data_in.size());
    long lock_span = ceil_xdivy(data_out.size(), locks.size());
    #pragma omp parallel for schedule(dynamic, TASKSIZE) shared(locks)
        for(int i = 0; i < n; i++){
            int idx = data_index[i];
            int lock_idx = idx/lock_span;
            omp_set_lock(&locks[lock_idx]);
            data_out[idx] += data_in[i];
            omp_unset_lock(&locks[lock_idx]);
            // Needs to release lock immediately due to data access indirections
        }
    return;
}

/**
 * A reduction based scatter reduce
 * @param scratch - a  <num_proc> * <data_out.size()> temporary matrix for thread
 *                  local aggregation
*/
void reduction_scatter_reduce(std::vector<double> &data_in,
                              std::vector<int> &data_index,
                              std::vector<double> &data_out,
                              std::vector<std::vector<double>> &scratch){
    int n = static_cast<int>(data_in.size());
    int n_proc = static_cast<int>(scratch.size());
    #pragma omp parallel // Local Aggregation Step
    {
        int thread_id = omp_get_thread_num();
        int idx;
        #pragma omp for schedule(dynamic, TASKSIZE)
            for(int i = 0; i < n; i++){
                idx = data_index[i];
                scratch[thread_id][idx] += data_in[i];
            }
    }
    // Across Thread Accumulation Phase. SIMD is not very useful here...
    int m = static_cast<int>(data_out.size());
    #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i < m; i += SIMDWIDTH){
            // #pragma omp simd
            for(int idx = 0; idx < SIMDWIDTH; idx++){
                for(int t = 0; t < n_proc; t ++)
                data_out[idx + i] += scratch[t][idx + i];
            }
        }
}






/**
 * Main routine
*/
int main(int argc, char **argv) {
    // Initialization Stage
    int num_threads = 0;
    long input_dim = 0L;
    long output_dim = 0L;
    long num_locks = 0L;
    char mode = '\0';
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "i:o:n:m:l:")) != -1) {
        switch (opt) {
        case 'i':
            input_dim = atol(optarg);
            break;
        case 'o':
            output_dim = atol(optarg);
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        case 'm':
            mode = *optarg;
            break;
        case 'l':
            num_locks = atol(optarg);
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }
    if(input_dim <= 0 || output_dim <= 0 || num_threads <= 0){
        usage(argv[0]);
        exit(1);
    }
    if(mode == 'L' && num_locks <= 0){
        std::cerr << "Lock Mode: Number of locks is invalid.\n";
        exit(1);
    }
        if(mode == 'L' && num_locks > std::min(1024, static_cast<int>(input_dim))){
        std::cerr << "Too many locks.\n";
        exit(1);
    }

    // Generate random Data
    std::vector<double> data_in(input_dim);
    std::vector<int> data_index(input_dim);
    for (long i = 0; i < input_dim; i++){
        data_in[i] = f_rand(1e-4, 1.0f);
        data_index[i] = i_rand(0, static_cast<int>(output_dim));
    }

    // Initialize Parallel Constructs
    omp_set_num_threads(num_threads);
    std::vector<std::vector<double>> scratch(num_threads, std::vector<double>(output_dim, 0.0f));
    std::vector<omp_lock_t>locks(num_locks);
    for(auto l:locks){
        omp_init_lock(&l);
    }
    std::vector<double> serial_out(output_dim, 0.0f);
    std::vector<double> parallel_out(output_dim, 0.0f);

    auto serial_start = timestamp;

    // Test Serial Implementation
    scatter_reduce(data_index, data_in, serial_out);

    auto serial_end = timestamp;

    auto parallel_start = timestamp;

    // Test Parallel Implementation
    if(mode == 'L'){ // lock
        lock_scatter_reduce(data_in, data_index, parallel_out, locks);
    }
    if(mode == 'R'){ // Reduction Based Scatter Reduce
        reduction_scatter_reduce(data_in, data_index, parallel_out, scratch);
    }

    auto parallel_end = timestamp;

    for(auto l:locks){
        omp_destroy_lock(&l);
    }
    std::cout << "serial computation time:   " << TIME(serial_start, serial_end) << "\n";
    std::cout << "parallel computation time: " << TIME(parallel_start, parallel_end) << "\n";
    std::cout << "Paralle Speedup:" << TIME(serial_start, serial_end) / TIME(parallel_start, parallel_end) << "\n";

    for(long i = 0; i < output_dim; i++){
        if(!is_zero(serial_out[i] -parallel_out[i])){
            printf("Parallel Solution Mismatch at [%d]: Expected %f, Actual: %f\n", i, serial_out[i], parallel_out[i]);
        }
    }
}