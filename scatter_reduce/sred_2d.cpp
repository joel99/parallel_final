/**
 * Testing file for different implementations of scatter reduce
 * Reflect the wordle challenge better by considering a 2D index to scatter of a 1D prior.
 * 2D index is Guess x Candidate (for simplicity, assume square = data dim, and scattered prior weight is also 1D = data dim)
 * Output dim is analogous to colors/target bins
 * Consider: Cache-advantage of reduction based multi-round, and lock protected round robin writes.
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
    // Expecting data to be square, prior to be arbitrary weights of length
    std::cout << "Usage:\n" << exec_name << "-i <data dimension> -o <output dimension> -n <thread count> -m <mode> [-l <number of locks>]\n";
    std::cout << "-m: 'L': lock based implementation, 'R': reduction based implementation, 'S': Serial\n";
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

// Serial
void scatter_reduce(std::vector<std::vector<int>> &index, 
                    std::vector<double> &in,
                    std::vector<std::vector<double>> &out){
    size_t guesses = index.size();
    size_t candidates = index[0].size();
    int idx;
    for(size_t guess = 0; guess < guesses; guess++){
        for (size_t cand = 0; cand < candidates; cand++){
            idx = index[guess][cand];
            out[guess][idx] += in[cand];
        }
    }
}

/**
 * A reduction based scatter reduce
 * @param scratch - a  <num_proc> * <data_out.size()> temporary matrix for thread
 *                  local aggregation (better than local allocation)
*/
void reduction_scatter_reduce(std::vector<double> &data_in, // input
                              std::vector<std::vector<int>> &data_index, // output by input
                              std::vector<std::vector<double>> &data_out, // input by color/output 
                              std::vector<std::vector<std::vector<double>>> &scratch){ // thread by input by color/output
    int guesses = static_cast<int>(data_index.size());
    int colors = static_cast<int>(data_out[0].size());
    std::cout << "Guesses: " << guesses << " Colors: " << colors << "\n";

    // Manual
    #pragma omp parallel // Local Aggregation Step
    {
        int thread_id = omp_get_thread_num();
        int idx;
        for (int guess = 0; guess < guesses; guess++){
            #pragma omp for
            for(int candidate = 0; candidate < data_index[guess].size(); candidate++){
                idx = data_index[guess][candidate];
                scratch[thread_id][guess][idx] += data_in[candidate];
            }
            // Hm.. do I accumulate across guess? That doesn't seem ok
            #pragma omp critical
            {
                for(int color = 0; color < colors; color++){
                    data_out[guess][color] += scratch[thread_id][guess][color];
                }
            }

        }
    }
}

// void reduction_scatter_reduce_omp(std::vector<double> &data_in,
//                               std::vector<std::vector<int>> &data_index,
//                               std::vector<std::vector<double>> &data_out){
//     int n = static_cast<int>(data_in.size());
//     int m = static_cast<int>(data_out.size());

//     double* data_out_ptr = data_out.data();
//     #pragma omp parallel
//     {
//         int idx;
//         #pragma omp for reduction(+:data_out_ptr[:m])
//         for(int i = 0; i < n; i++){
//             idx = data_index[i];
//             data_out_ptr[idx] += data_in[i];
//         }
//     }
// }

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
    unsigned int seed = 0; // Seed for random number generator
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "i:o:n:m:l:s:")) != -1) {
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
        case 's': // Handle seed parameter
            seed = static_cast<unsigned int>(atoi(optarg));
            srand(seed); // Seed the random number generator
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
    // 2D index to reflect proper data
    std::vector<std::vector<int>> data_index(input_dim, std::vector<int>(input_dim));
    for (long i = 0; i < input_dim; i++){
        data_in[i] = f_rand(1e-4, 1.0f);
        for (long j = 0; j < input_dim; j++){
            data_index[i][j] = i_rand(0, static_cast<int>(output_dim));
        }
    }

    // Initialize Parallel Constructs
    omp_set_num_threads(num_threads);
    std::vector<std::vector<std::vector<double>>> scratch(num_threads, std::vector<std::vector<double>>(input_dim, std::vector<double>(output_dim, 0.0)));

    // std::vector<std::vector<std::vector<double>>> scratch(num_threads, std::vector<double>(input_dim, std::vector<double>(output_dim, 0.0f)));
    std::vector<omp_lock_t>locks(num_locks);
    for(int i = 0; i < num_locks; i++){
        omp_init_lock(&locks[i]);
    }
    std::vector<std::vector<double>> serial_out(input_dim, std::vector<double>(output_dim, 0.0f));
    std::vector<std::vector<double>> parallel_out(input_dim, std::vector<double>(output_dim, 0.0f));

    auto serial_start = timestamp;

    // Test Serial Implementation
    scatter_reduce(data_index, data_in, serial_out);

    auto serial_end = timestamp;

    auto parallel_start = timestamp;

    // Test Parallel Implementation
    // if(mode == 'L'){ // lock
    //     lock_scatter_reduce(data_in, data_index, parallel_out, locks);
    // }
    if(mode == 'R'){ // Reduction Based Scatter Reduce
        reduction_scatter_reduce(data_in, data_index, parallel_out, scratch);
    }
    // if (mode == 'M') {
    //     reduction_scatter_reduce_omp(data_in, data_index, parallel_out);
    // }

    auto parallel_end = timestamp;

    for(auto l:locks){
        omp_destroy_lock(&l);
    }
    std::cout << "serial computation time:   " << TIME(serial_start, serial_end) << "\n";
    std::cout << "parallel computation time: " << TIME(parallel_start, parallel_end) << "\n";
    std::cout << "Parallel Speedup:" << TIME(serial_start, serial_end) / TIME(parallel_start, parallel_end) << "\n";

    for(long i = 0; i < output_dim; i++){
        for (long j = 0; j < input_dim; j++){
            if(!is_zero(serial_out[j][i] -parallel_out[j][i])){
                printf("Parallel Solution Mismatch at [%lu][%lu]: Expected %f, Actual: %f\n", j, i, serial_out[j][i], parallel_out[j][i]);
            }
        }
    }
}