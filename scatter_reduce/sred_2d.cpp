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
#include <list>

#include <omp.h>

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
#define TASKSIZE 32
#define SIMDWIDTH 32
#define PRECISION 1e-4

void usage(char *exec_name){
    // Expecting data to be square, prior to be arbitrary weights of length
    // -i guess/candidate
    // -o colors
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
*/
void reduction_scatter_reduce(std::vector<double> &data_in, // input
                              std::vector<std::vector<int>> &data_index, // output by input
                              std::vector<std::vector<double>> &data_out, // input by color/output 
                              std::vector<std::vector<std::vector<double>>> &scratch){ // thread by input by color/output
    // Having #omp fors inside loops impose untenable overhead, do manual computation (e.g. 0.5x to 2.5x speedup for 8 threads)
    int guesses = static_cast<int>(data_index.size());
    int colors = static_cast<int>(data_out[0].size());
    std::cout << "Guesses: " << guesses << " Colors: " << colors << "\n";

    // Manual inner assignment is killer, explicitly compute work allocation
    #pragma omp parallel // Local Aggregation Step
    {
        int thread_id = omp_get_thread_num();
        int idx;
        int candidate_span = ceil_xdivy(guesses, omp_get_num_threads());
        int read_min = candidate_span * thread_id;
        int read_max = std::min(read_min + candidate_span, guesses);
        for (int guess = 0; guess < guesses; guess++){
            // #pragma omp for
            // for(int candidate = 0; candidate < data_index[guess].size(); candidate++){
            for(int candidate = read_min; candidate < read_max; candidate++){
                idx = data_index[guess][candidate];
                scratch[thread_id][guess][idx] += data_in[candidate];
            }
            #pragma omp critical
            {
                for(int color = 0; color < colors; color++){
                    data_out[guess][color] += scratch[thread_id][guess][color];
                }
            }
        }
        // No alloc, theoretical near equivalent to OMP
        // std::vector<double> local_sum(colors, 0.0); // Each thread has a local sum array
        // int idx;

        // for (int guess = 0; guess < guesses; guess++) {
        //     local_sum.assign(colors, 0.0); // Reset local sum
        //     #pragma omp for
        //     for (int candidate = 0; candidate < data_index[guess].size(); candidate++) {
        //         idx = data_index[guess][candidate];
        //         local_sum[idx] += data_in[candidate]; // Accumulate locally
        //     }
        //     #pragma omp critical // Safely add local sums to the main array
        //     for (int i = 0; i < colors; i++) {
        //         data_out_flat[guess * colors + i] += local_sum[i];
        //     }
        // }
    }
}

void reduction_scatter_reduce_omp(std::vector<double> &data_in, // input
                              std::vector<std::vector<int>> &data_index, // guess by inputs
                            //   std::vector<std::vector<double>> &data_out){ // guess by color/output 
                              std::vector<double> &data_out_flat){ // guess by color/output 
    int guesses = static_cast<int>(data_index.size());
    int colors = static_cast<int>(data_out_flat.size() / guesses);
    // std::cout << "Data in Size: " << data_in.size() << "\n";
    // std::cout << "Data Index Size: " << data_index.size() << "x" << data_index[0].size() <<  "\n";
    // std::cout << "Guesses: " << guesses << " Colors: " << colors << "\n";
    // std::cout << "Data Out Size: " << data_out_flat.size() << "\n";
    double* data_out_ptr = data_out_flat.data();
    // Manual
    #pragma omp parallel // Local Aggregation Step
    {
        // *OMP - fails for longer lengths (IO shape 12800 x 256, i.e. standard length, 1x263609) for unknown reason, manual above still works
        int idx;
        for (int guess = 0; guess < guesses; guess++){
            #pragma omp for reduction(+:data_out_ptr[guess * colors:colors]) // syntax is start:length
            for(int candidate = 0; candidate < data_index[guess].size(); candidate++){
                idx = data_index[guess][candidate];
                data_out_ptr[guess * colors + idx] += data_in[candidate];
            }
        }
    }
}

void reduction_scatter_reduce_pipeline(std::vector<double> &data_in, // input
                              std::vector<std::vector<int>> &data_index, // output by input
                              std::vector<std::vector<double>> &data_out, // input by color/output 
                              std::vector<std::vector<std::vector<double>>> &scratch,
                              std::vector<omp_lock_t> &locks){ // thread by input by color/output
    /*
        Hybrid guess-candidate parallel, to demonstrate a point about candidate
        Here threads will track and be adding and attempting to reduce work queues accumulated across guesses.

        TODO illustrate memory consumption of full scratch
    */
    int guesses = static_cast<int>(data_index.size());
    int colors = static_cast<int>(data_out[0].size());

    int num_threads = omp_get_max_threads();
    // TODO, there's no reason only own thread should commit own work other than cache effect
    // TODO profile write/read ratio
    // thread by guesses to process, list because we will be popping
    std::vector<std::list<int>> task_queue = std::vector<std::list<int>>(num_threads, std::list<int>());

    // Hypothetical gains over guess-parallel if shared cache can be leveraged
    // Manual
    int candidate_span = ceil_xdivy(guesses, num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int read_min = candidate_span * thread_id;
        int read_max = std::min(read_min + candidate_span, guesses);

        int idx;
        for (int guess = 0; guess < guesses; guess++){
            for(int candidate = read_min; candidate < read_max; candidate++){
                idx = data_index[guess][candidate];
                scratch[thread_id][guess][idx] += data_in[candidate];
            }
            task_queue[thread_id].push_back(guess);
            // attempt to clear accumulated work, iterate through list
            auto it = task_queue[thread_id].begin();
            while (it != task_queue[thread_id].end()) {
                int write_guess = *it;
                if (omp_test_lock(&locks[write_guess])) {
                    for (int color = 0; color < colors; color++) {
                        data_out[write_guess][color] += scratch[thread_id][write_guess][color];
                    }
                    omp_unset_lock(&locks[write_guess]);
                    it = task_queue[thread_id].erase(it); // Erase returns the next iterator
                } else {
                    ++it;
                    // Only move to next element if task wasn't done (i.e., lock not acquired)
                }
            }
        }
        // clear queues - only hit on large problems
        auto it = task_queue[thread_id].begin();
        while (it != task_queue[thread_id].end()) {
            int write_guess = *it;
            std::cout << "Thread: " << thread_id << " Writing Guess: " << write_guess << "\n";
            omp_set_lock(&locks[write_guess]);
            for (int color = 0; color < colors; color++) {
                data_out[write_guess][color] += scratch[thread_id][write_guess][color];
            }
            omp_unset_lock(&locks[write_guess]);
            ++it;
        }
    }
}

void reduction_scatter_reduce_cap(std::vector<double> &data_in, // input
                              std::vector<std::vector<int>> &data_index, // output by input
                              std::vector<std::vector<double>> &data_out, // input by color/output 
                              std::vector<std::vector<std::vector<double>>> &scratch, // capacity x color/output
                              std::vector<omp_lock_t> &locks){ // thread by input by color/output
    /*
        Hybrid guess-candidate parallel, to demonstrate a point about candidate
        Here threads will track and be adding and attempting to reduce work queues accumulated across guesses.

        This version uses limited capacity scratch, so we cannot have arbitrarily large queues.
    */
    int guesses = static_cast<int>(data_index.size());
    int colors = static_cast<int>(data_out[0].size());
    int capacity = static_cast<int>(scratch[0].size());
    std::cout << "Guesses: " << guesses << " Colors: " << colors << " Capacity: " << capacity << "\n";

    int num_threads = omp_get_max_threads();
    // TODO, there's no reason only own thread should commit own work other than cache effect
    // TODO profile write/read ratio
    // thread by capacity to process, true if there's data in scratch to write
    std::vector<std::vector<bool>> task_mask = std::vector<std::vector<bool>>(num_threads, std::vector<bool>(capacity, false));
    // thread by work-item of (lane, guess) pairs to know what to map from/to
    auto task_queue = std::vector<std::list<std::pair<int, int>>>(
        num_threads, std::list<std::pair<int, int>>());

    // Hypothetical gains over guess-parallel if shared cache can be leveraged
    // Manual
    int candidate_span = ceil_xdivy(guesses, num_threads);
    auto scatter_start = timestamp;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int read_min = candidate_span * thread_id;
        int read_max = std::min(read_min + candidate_span, guesses);

        int idx;
        #pragma omp single
        scatter_start = timestamp;

        for (int guess = 0; guess < guesses; guess++){
            // find the next uncommitted slot in scratch, and stage data
            int write_lane = -1;
            for (int i = 0; i < capacity; i++){
                if(!task_mask[thread_id][i]){
                    write_lane = i; 
                    scratch[thread_id][write_lane].assign(colors, 0.0); // clear
                    task_mask[thread_id][write_lane] = true;
                    break;
                }
            }
            // std::cout << "Thread: " << thread_id << " Writing Lane: " << write_lane << "\n";
            for(int candidate = read_min; candidate < read_max; candidate++){
                idx = data_index[guess][candidate];
                scratch[thread_id][write_lane][idx] += data_in[candidate];
            }
            task_queue[thread_id].push_back(std::make_pair(guess, write_lane));
            bool try_once = true;
            // std::cout << "Thread: " << thread_id << " Queue Size: " << task_queue[thread_id].size() << "\n";
            // attempt to clear accumulated work, iterate through list, and do not exceed capacity
            while (try_once || task_queue[thread_id].size() >= capacity) {
                auto it = task_queue[thread_id].begin();
                while (it != task_queue[thread_id].end()) {
                    int write_guess = it->first;
                    int write_lane = it->second;
                    // std::cout << "Thread: " << thread_id << " Guess / Lane " << write_guess << " / " << write_lane << "\n";
                    if (omp_test_lock(&locks[write_guess])) {
                        for (int color = 0; color < colors; color++) {
                            // std::cout << "Thread: " << thread_id << " color / lane " << color << " / " << write_lane << "\n";
                            data_out[write_guess][color] += scratch[thread_id][write_lane][color];
                        }
                        omp_unset_lock(&locks[write_guess]);
                        // std::cout << "Thread: " << thread_id << " Wrote Guess - free now: " << write_guess << "\n";
                        task_mask[thread_id][write_lane] = false;
                        it = task_queue[thread_id].erase(it); // Erase returns the next iterator
                    } else {
                        ++it;
                        // Only move to next element if task wasn't done (i.e., lock not acquired)
                    }
                }
                try_once = false;
            }
        }
        // #pragma omp single
        // std::cout << "Thread: " << thread_id << " Clearing now \n";
        // clear queues - TODO assess how much is used here
        auto it = task_queue[thread_id].begin();
        while (it != task_queue[thread_id].end()) {
            auto pair = task_queue[thread_id].front();
            int write_guess = pair.first;
            int write_lane = pair.second;
            std::cout << "Thread: " << thread_id << " Writing Guess: " << write_guess << "\n";
            omp_set_lock(&locks[write_guess]);
            for (int color = 0; color < colors; color++) {
                data_out[write_guess][color] += scratch[thread_id][write_lane][color];
            }
            omp_unset_lock(&locks[write_guess]);
            ++it;
        }
    }
    auto scatter_end = timestamp;
    std::cout << "Scatter Time: " << TIME(scatter_start, scatter_end) << "\n";
}

void reduction_scatter_reduce_row(std::vector<double> &data_in, // input
                              std::vector<std::vector<int>> &data_index, // output by input
                              std::vector<std::vector<double>> &data_out // input by color/output 
                            ){
    /*
        Guess-parallel
        This version uses limited capacity scratch, so we cannot have arbitrarily large queues.
    */
    int guesses = static_cast<int>(data_index.size());
    int candidates = static_cast<int>(data_index[0].size());
    int colors = static_cast<int>(data_out[0].size());
    std::cout << "Guesses: " << guesses << " Colors: " << colors << " \n";

    #pragma omp parallel
    {
        int idx;
        #pragma omp for nowait
        for (int guess = 0; guess < guesses; guess++){
            for(int candidate = 0; candidate < candidates; candidate++){
                idx = data_index[guess][candidate];
                data_out[guess][idx] += data_in[candidate];
            }
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
    unsigned int seed = 0; // Seed for random number generator
    int opt;
    int capacity = 1; // Scratch capacity
    // Read program parameters
    while ((opt = getopt(argc, argv, "i:o:c:n:m:l:s:")) != -1) {
        switch (opt) {
        case 'i':
            input_dim = atol(optarg);
            break;
        case 'o':
            output_dim = atol(optarg);
            break;
        case 'c':
            capacity = atoi(optarg);
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

    std::vector<std::vector<std::vector<double>>> scratch;
    if (mode == 'P' || mode == 'R' || mode == 'C') {
        int scratch_cap = input_dim;
        if (mode == 'C') {
            scratch_cap = capacity;
        }
        scratch = std::vector<std::vector<std::vector<double>>>(num_threads, std::vector<std::vector<double>>(scratch_cap, std::vector<double>(output_dim, 0.0)));
    }

    // std::vector<std::vector<std::vector<double>>> scratch(num_threads, std::vector<double>(input_dim, std::vector<double>(output_dim, 0.0f)));
    std::vector<omp_lock_t> locks;
    if (mode == 'P' || mode == 'C') {
        int guesses = static_cast<int>(data_index.size());
        locks = std::vector<omp_lock_t>(guesses); 
        // init locks
        for(int i = 0; i < guesses; i++){
            omp_init_lock(&locks[i]);
        }
    } else {
        locks = std::vector<omp_lock_t>(num_locks);
        for(int i = 0; i < num_locks; i++){
            omp_init_lock(&locks[i]);
        }
    }
    std::vector<std::vector<double>> serial_out(input_dim, std::vector<double>(output_dim, 0.0f));
    std::vector<std::vector<double>> parallel_out;
    std::vector<double> parallel_out_flat;
    if (mode != 'S') {
        parallel_out = std::vector<std::vector<double>>(input_dim, std::vector<double>(output_dim, 0.0f));
    }
    if (mode == 'M') {
        parallel_out_flat = std::vector<double>(input_dim * output_dim, 0.0f);
    }

    // Print the inputs
    // for (long i = 0; i < input_dim; i++){
    //     for (long j = 0; j < input_dim; j++){
    //         printf("%d ", data_index[i][j]);
    //     }
    //     printf("\n");
    // }

    // for (long i = 0; i < input_dim; i++){
    //     printf("%f ", data_in[i]);
    // }

    std::cout << "Starting...\n";
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
    if (mode == 'M') {
        // reduction_scatter_reduce_omp(data_in, data_index, parallel_out);
        reduction_scatter_reduce_omp(data_in, data_index, parallel_out_flat);
    }
    if (mode == 'P') {
        reduction_scatter_reduce_pipeline(data_in, data_index, parallel_out, scratch, locks);
    }
    if (mode == 'C') {
        reduction_scatter_reduce_cap(data_in, data_index, parallel_out, scratch, locks);
    }
    if (mode == 'G') {
        reduction_scatter_reduce_row(data_in, data_index, parallel_out);
    }

    auto parallel_end = timestamp;

    // Teardown
    if (mode == 'M') {
        // write back in
        for (long i = 0; i < input_dim; i++){
            for (long j = 0; j < output_dim; j++){
                parallel_out[i][j] = parallel_out_flat[i * output_dim + j];
            }
        }
    }

    for(auto l:locks){
        omp_destroy_lock(&l);
    }
    std::cout << "serial computation time:   " << TIME(serial_start, serial_end) << "\n";
    if (mode != 'S') {
        std::cout << "parallel computation time: " << TIME(parallel_start, parallel_end) << "\n";
        std::cout << "Parallel Speedup:" << TIME(serial_start, serial_end) / TIME(parallel_start, parallel_end) << "\n";
        for(long i = 0; i < output_dim; i++){
            for (long j = 0; j < input_dim; j++){
                if(!is_zero(serial_out[j][i] -parallel_out[j][i])){
                    printf("Parallel Solution Mismatch at [%lu][%lu]: Expected %f, Actual: %f\n", j, i, serial_out[j][i], parallel_out[j][i]);
                    exit(1);
                }
            }
        }
    }
}