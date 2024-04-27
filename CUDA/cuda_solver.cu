#include <string>
#include <algorithm>
#include <math.h>
#include <vector>
#include <stdio.h>
#include <chrono>


#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "word.h"
#include "cuda_solver.h"
#include "cuda_utils.h"

/**********************
 * Math and Auxillary Functions
***********************/

#define MAXITERS 10

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()

// Cuda Error Checking Function from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ float prior_to_entropy(float prior, float normalize_factor){
    if(fabsf(prior) <= 1e-12) return 0.0f;
    float prob = prior/normalize_factor;
    return (-1.0f) * prob * log2f(prob);
}



int ceil_xdivy_int(int X, int Y){
    return (X + (Y - 1)) / Y;
}

/**********************
 * Some CUDA constants
***********************/

// Pre-computation phase constants
#define __MATRIX_BLOCKDIM__  32 // CUDA limits a maximum of 1024 threads per block...
#define __MATRIX_BLOCKSIZE__ (__MATRIX_BLOCKDIM__ * __MATRIX_BLOCKDIM__)

// Solver phase constants
// The number of threads per block in the computation phase (128 seems to be good)
#define __COMPUTE_BLOCKDIM__ 128
// the number of patterns (6561) Rounded to the nearest multiple of BLOCKDIM
#define __NUM_PATTERN_ALLOC__ 6656 
// Allocation size for additional scratch work. 4096 by default.
#define __CACHE_ALLOC__ 4096
// The number of elements to be processed in one pass. Must be a multiple of BLOCKDIM
#define __COMPUTE_STRIDE__ 2560


/**
 * Global Constants for easy of access within kernels
 * @param wordlist A pointer to the start of the word list in device memory
 * @param priorlist A pointer to the start of the prior list in device memory
 * @param pattern_matrix A pointer to the start of the pattern_matrix
 * @param scores A pointer to the start of the score list (in solver routine)
 * @param num_words The number of words in the word list
 * @param prior_sum A pointer to the sum of all prior weights.
 * @param remaining A pointer to the number of words remaining in the solver
 * @param feedback A pointer to the feedback obtained from the current guess
 * @param candidate A pointer to the candidate word chosen.
*/
typedef struct GlobalConstants{
    // Global Constants
    size_t num_words;
    int wordlen;
    float initial_prior_sum;
    // Pointers to Key Data Structures
    word_t *wordlist;
    float *priorlist;
    float *prior_sum;
    coloring_t *pattern_matrix;
    // Temporary Variables for Solver
    float *scores;
    int *remaining;
    coloring_t *feedback;
    int *candidate;
    // Pointers to solver results
    // int *game_rounds;
    // int *game_solved;
    
} GlobalConstants;

__constant__ GlobalConstants CudaParams;

/****************
 * CUDA kernels
*****************/

// Helper kernels

__device__ int index_convert(int query_index, int answer_index, int num_words){
    return (query_index * num_words) + answer_index;
}

__device__ coloring_t cuda_word_cmp(word_t &query, word_t &answer, int wordlen){
    // Optimized with some 15-213 data lab magic
    int query_matched = 0x0;
    int mask = 0x1;

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for(int i = 0; i < wordlen; i++){
        // printf("%x, %x\n", mask, query_matched);
        if(query.text[i] == answer.text[i]){
            out += (2 * mult);
            query_matched |= mask;
        }
        mask <<= 1;
        mult *= NUMCOLORS;
    }

    // reset multiplier and mask
    mask = 0x1;
    mult = 1;
    // Add a new lane mask indicating if answer is matched
    int answer_matched = query_matched;
    // Check for yellow boxes
    for(int i = 0; i < wordlen; i++){ // query index
        if(query_matched & mask) {
            mult *= NUMCOLORS;
            mask <<= 1;
            continue;
        }
        int answer_mask = 0x1;
        for(int j = 0; j < wordlen; j++){// answer index
            // printf("%x, %x, %x\n", answer_mask, answer_matched, answer_matched & answer_mask);
            if(!(answer_matched & answer_mask) && query.text[i] == answer.text[j]){
                out += mult;
                query_matched |= mask;
                answer_matched |= answer_mask;
                break;
            }
            answer_mask <<= 1;
        }
        mult *= NUMCOLORS;
        mask <<= 1;
    }
    return out;
}

__global__ void pattern_compute_main(void){
    // Initialize shared cache
    __shared__ word_t queries[__MATRIX_BLOCKDIM__];
    __shared__ word_t answers[__MATRIX_BLOCKDIM__];

    // Assign the indicies of each worker thread:
    int idx_q = blockIdx.y * blockDim.y + threadIdx.y; // vertical / query axis
    int idx_a = blockIdx.x * blockDim.x + threadIdx.x; // horizontal / answer axis
    int copy_idx;

    int num_words = static_cast<int> (CudaParams.num_words);
    coloring_t *matrix = CudaParams.pattern_matrix;
    if(threadIdx.y == 0){
        // Copy the queries (careful about out-of-bounds)
        copy_idx = blockIdx.y * blockDim.y + threadIdx.x;
        if(copy_idx < num_words)
            queries[threadIdx.x] = CudaParams.wordlist[copy_idx];
    }
    if(threadIdx.y == 1){
        // Copy the answers
        copy_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(copy_idx < num_words)
            answers[threadIdx.x] = CudaParams.wordlist[copy_idx];
    }

    __syncthreads();
    coloring_t pattern;
    int max_wordlen = CudaParams.wordlen;
    // Now queries and answers are copied into the cached memory. Now perform computation.
    if(idx_q < num_words && idx_a < num_words){
        pattern = cuda_word_cmp(queries[threadIdx.y], answers[threadIdx.x], max_wordlen);
        matrix[index_convert(idx_q, idx_a, num_words)] = pattern;
    }
}


/**
 * Helper routine: determines the lower bound and the upper bound (excl.)
 * of a statically assigned task
*/
__device__ __inline__ void get_range(int total, int numThreads, int tid,
    int *start, int *end){
    int avg_count = total / numThreads;
    int remainder = total % numThreads;
    int task_size = (tid < remainder) ? avg_count + 1 : avg_count;
    *start = (tid < remainder) ? (avg_count + 1) * tid : (avg_count * tid) + remainder;
    *end = *start + task_size;
}

/**
 * Within-Block helper function: Performs the parallel part of a reduction
 * @param in The input array
 * @param out The buffer where each thread will post its local aggregation result
 * @param N The dimension of the input array
 * @warning out should have at least sizeof(float) * blockDim.x bytes
*/
__device__ __inline__ void within_block_reduce(float *in, float *out, int N){
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x){
        local_sum += in[i];
    }
    out[threadIdx.x] = local_sum;
}


/**
 * Subroutine of compute_main: Reads a chunk of the data into the cache
 * And perform scatter reduce on that chunk of the data
 * @warning requires that chunkSize (ie. __COMPUTE_STRIDE__) is a multiple of numThreads.
 * @warning scatter_reduce_out should be in device shared memory.
*/
__device__ __inline__ void read_chunk_scatter_reduce(int num_words, int num_patterns, 
    int start_index, int *cache, float *scatter_reduce_out){
    const int numThreads = blockDim.x; // 1D threadblock
    const int chunkSize = __COMPUTE_STRIDE__;
    const int tid = threadIdx.x;
    // Distribute the cache work space
    float *cache_input = (float*)cache;
    coloring_t *cache_index = (coloring_t*)&cache_input[chunkSize];
    // Compute where this thread block should be reading from
    float *priors_read_from = CudaParams.priorlist;
    coloring_t *patterns_read_from = &((CudaParams.pattern_matrix)[blockIdx.x * num_words]);

    // First copy the global data into the cache
    int global_index;
    for(int local_index = tid; local_index < chunkSize; local_index += numThreads){
        global_index = start_index + local_index;
        if(global_index >= num_words){ // Prevent some uninitialised memory issues;
            cache_input[local_index] = 0.0f;
            cache_index[local_index] = 0;
        }
        else{
            cache_input[local_index] = priors_read_from[global_index];
            cache_index[local_index] = patterns_read_from[global_index];
        }
    }
    __syncthreads();
    // Now compute the index bounds 
    int start;
    int end;
    get_range(num_patterns, numThreads, tid, &start, &end);
    // Perform scatter reduce locally
    for(int i = 0; i < chunkSize; i++){
        int j = static_cast<int>(cache_index[i]);
        if(start <= j && j < end){
            scatter_reduce_out[j] += cache_input[i];
        }
    }
    // __syncthreads(); at caller code
}

/**
 * Main routine for score computation. Each block should work on a single row
 * of the pattern matrix.
 * @note blockIdx.x -> candidate / row index in the CPU solver
 * @note threadIdx.x -> Answer / Column index in the CPU solver
*/
__global__ void score_compute_main(int num_patterns){
    // Initialization of shared cache memory
    __shared__ float scratch[__NUM_PATTERN_ALLOC__];
    __shared__ int cache[__CACHE_ALLOC__];
    // Need to make sure that the scratch array is zero initialized
    for (int i = threadIdx.x; i < __NUM_PATTERN_ALLOC__; i += blockDim.x) 
        scratch[i] = 0.0f;
    int num_words = static_cast<int>(CudaParams.num_words);
    float prior_sum = *(CudaParams.prior_sum);
    // Process the scatter reduce array by chunk
    for(int start_idx = 0; start_idx < num_words; start_idx += __COMPUTE_STRIDE__){
        read_chunk_scatter_reduce(num_words , num_patterns, start_idx, cache, scratch);
        __syncthreads();
    }
    // Now the scratch array should be populated with the pooled up weights
    for(int i = threadIdx.x; i < num_patterns; i += blockDim.x)
        scratch[i] = prior_to_entropy(scratch[i], prior_sum);
    __syncthreads();
    // Perform reduction on the scratch array that is populated with the entropies
    float *tmp_reduce = (float*)cache; // Repurpose the scatter reduce cache.
    within_block_reduce(scratch, tmp_reduce, num_patterns);
    __syncthreads();
    // Serial code here. Thread 0 will post the reduction result to global scores
    float score = 0.0f;
    if(threadIdx.x == 0){
        for(int i = 0; i < blockDim.x; i++){
            score += tmp_reduce[i];
        }
        score += ((CudaParams.priorlist)[blockIdx.x]) / prior_sum; // Finally add the prior weight
        (CudaParams.scores)[blockIdx.x] = score;
    }
}

/**
 * Given that the candidate has already been elected, store the feedback 
 * corresponding to a particular answer
 * @warning Kernel only executed by one thread
*/
__global__ void get_candidate_feedback(int answer_idx){
    if(threadIdx.x == 0){
        int num_words = static_cast<int>(CudaParams.num_words);
        int candidate = *(CudaParams.candidate);
        coloring_t *matrix = (CudaParams.pattern_matrix);
        coloring_t feedback = matrix[num_words * candidate + answer_idx];
        // Finally store the feedback into the device memory.
        *(CudaParams.feedback) = feedback;
    }
}

/**
 * Kernel for updating the prior list
*/
__global__ void update_prior(void){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_words = static_cast<int>(CudaParams.num_words);
    if(tid >= num_words) return; // Prevent out of bound addressing.

    // Record the row of the pattern matrix
    coloring_t *row = &(
        CudaParams.pattern_matrix[ num_words * static_cast<int>(*(CudaParams.candidate))]);
    coloring_t feedback = *(CudaParams.feedback);
    float *priors = CudaParams.priorlist;

    // Zero out all answers which the feedback do not match with the actual answer.
    if(row[tid] != feedback){
        priors[tid] = 0.0f;
    }
}




/************************************************
 * CUDA Solver (CPU Code)
*************************************************/

/** 
 * Constructor for the CudaSolver class
*/
CudaSolver::CudaSolver(){
    num_words = 0;
    prior_sum = 0.0f;
    _CPU_wordlist = NULL;
    _CPU_priorlist = NULL;

    _CUDA_wordlist = NULL;
    _CUDA_priorlist = NULL;
    _CUDA_word_scores = NULL;
    _CUDA_pattern_matrix = NULL;

    solver_prior_sum = NULL;
    solver_words_remaining = NULL;
    solver_guess_candidate = NULL;
    solver_candidate_score = NULL;
    solver_feedback = NULL;

    _CUDA_scratch_float = NULL;
    _CUDA_scratch_int = NULL;
}

/**
 * Destructor for the CudaSolver class
*/
CudaSolver::~CudaSolver(){
    if(_CPU_wordlist){
        delete [] _CPU_wordlist;
        delete [] _CPU_priorlist;
    }
    
    if (_CUDA_wordlist){
        cudaFree(_CUDA_wordlist);
        cudaFree(_CUDA_priorlist);
        cudaFree(_CUDA_word_scores);
        cudaFree(_CUDA_pattern_matrix);
        cudaFree(_CUDA_scratch_int);
        cudaFree(_CUDA_scratch_float);
    }

    if(solver_prior_sum){
        cudaFree(solver_prior_sum);
        cudaFree(solver_words_remaining);
        cudaFree(solver_guess_candidate);
        cudaFree(solver_candidate_score);
        cudaFree(solver_feedback);
    }
}

void CudaSolver::data_copy(std::vector<word_t> &word_list,
                           std::vector<float> &prior_list, 
                           float prior_sum_in){
    num_words = word_list.size();
    prior_sum = prior_sum_in;

    // Allocate space for CPU items
    _CPU_wordlist = new word_t[num_words];
    _CPU_priorlist = new float[num_words];

    std::copy(word_list.begin(), word_list.end(), _CPU_wordlist);
    std::copy(prior_list.begin(), prior_list.end(), _CPU_priorlist);
}

void CudaSolver::cuda_setup(){
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA solver\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
    
    if(!_CPU_wordlist || !_CPU_priorlist){
        fprintf(stderr, "cuda_setup called before CPU data fields are initialized!\n");
        return;
    }

    // Allocate device memory
    cudaMalloc(&_CUDA_wordlist, sizeof(word_t) * num_words);
    cudaMalloc(&_CUDA_priorlist, sizeof(float) * num_words);
    cudaMalloc(&_CUDA_word_scores, sizeof(float) * num_words);
    cudaMalloc(&_CUDA_pattern_matrix, sizeof(coloring_t) * num_words * num_words);

    // Allocate device memory for temporary solver variables
    cudaMalloc(&solver_prior_sum, sizeof(float));
    cudaMalloc(&solver_words_remaining, sizeof(int));
    cudaMalloc(&solver_guess_candidate, sizeof(int));
    cudaMalloc(&solver_candidate_score, sizeof(float));
    cudaMalloc(&solver_feedback, sizeof(coloring_t));

    // Allocate device memory for reduction scratch
    int reduction_scratch_size = ceil_xdivy_int(num_words, BLOCKDIM * 2);
    cudaMalloc(&_CUDA_scratch_int, reduction_scratch_size * sizeof(int));
    cudaMalloc(&_CUDA_scratch_float, reduction_scratch_size * sizeof(float));
    

    /**
     * The setup function only copies the wordlist into the device memory
     * The priorlist needs to be copied to the device memory during each solve
     * as the solver will destructively modify the prior list
    */
    cudaMemcpy(_CUDA_wordlist, _CPU_wordlist, sizeof(word_t) * num_words, cudaMemcpyHostToDevice);

    // Now initialize the global parameters so that kernels can have access to
    GlobalConstants temp;
    temp.wordlist = _CUDA_wordlist;
    temp.priorlist = _CUDA_priorlist;
    temp.pattern_matrix = _CUDA_pattern_matrix;
    temp.scores = _CUDA_word_scores;
    temp.num_words = num_words;
    temp.wordlen = wordlen;
    temp.initial_prior_sum = prior_sum;
    temp.prior_sum = solver_prior_sum;
    temp.remaining = solver_words_remaining;
    temp.feedback = solver_feedback;
    temp.candidate = solver_guess_candidate;

    cudaMemcpyToSymbol(CudaParams, &temp, sizeof(GlobalConstants));
}


void CudaSolver::pattern_comute(){
    dim3 blockDim(__MATRIX_BLOCKDIM__, __MATRIX_BLOCKDIM__, 1);
    dim3 gridDim(ceil_xdivy_int(num_words, blockDim.x), ceil_xdivy_int(num_words, blockDim.y));
    // Launch the precomputation kernel
    printf("Grid Size: (%d,%d); BlockSize: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    pattern_compute_main<<<gridDim, blockDim>>>();
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

bool CudaSolver::pattern_matrix_verify(){
    // Initialize the cpu patterm matrix
    coloring_t *CPU_out = new coloring_t[num_words * num_words];
    coloring_t *GPU_out = new coloring_t[num_words * num_words];

    for(int query_idx = 0; query_idx < num_words; query_idx++){
        word_t query = _CPU_wordlist[query_idx];
        for (int candidate_idx = 0; candidate_idx < num_words; candidate_idx++){
            CPU_out[query_idx * num_words + candidate_idx] = 
                word_cmp(query, _CPU_wordlist[candidate_idx]);
        }
    }
    // Copy the GPU result
    cudaMemcpy(GPU_out, _CUDA_pattern_matrix, 
        sizeof(coloring_t) * num_words * num_words,
        cudaMemcpyDeviceToHost);
    // Check if everything is correct
    bool out = true;
    for(int query_idx = 0; query_idx < num_words; query_idx++){
        for (int candidate_idx = 0; candidate_idx < num_words; candidate_idx++){
            if(CPU_out[query_idx * num_words + candidate_idx] != 
               GPU_out[query_idx * num_words + candidate_idx]){
                    std::cout << "CPU Result: ";
                    word_print(_CPU_wordlist[query_idx], CPU_out[query_idx * num_words + candidate_idx], ' ');
                    std::cout << "GPU Result: ";
                    word_print(_CPU_wordlist[query_idx], GPU_out[query_idx * num_words + candidate_idx], ' ');
                    std::cout << "Answer Word: ";
                    word_print(_CPU_wordlist[candidate_idx]);
                    out = false;
               }
        }
    }
    delete [] CPU_out;
    delete [] GPU_out;
    return out;
}

int CudaSolver::solve_verbose(int test_idx){
    auto setup_start = timestamp;

    // Initialize the solver temporary variables
    cudaMemcpy(_CUDA_priorlist, _CPU_priorlist, sizeof(float) * num_words, cudaMemcpyHostToDevice);
    cudaMemcpy(solver_prior_sum, &prior_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(solver_words_remaining, &num_words, sizeof(size_t), cudaMemcpyHostToDevice);
    // Keep track of the number of words remaining on the CPU side as well.
    int words_remaining = static_cast<int>(num_words);
    // Obtain the actual number of patterns to be passed into the kernel
    int num_patterns = get_num_patterns(); 
    int iterations = 0;
    coloring_t current_feedback;

    auto setup_end = timestamp;
     std::cout << "\nSetup Time: " << TIME(setup_start, setup_end) << "\n\n";

    while(iterations < MAXITERS){ // 10 iterations max to guarantee correctness

        auto compute_start = timestamp;

        if(words_remaining <= 2){
            // If the number of valid words remaining is at most 2, 
            // just perform random guess by taking the argmax
            cuda_max(_CUDA_priorlist, NULL, static_cast<int>(num_words), 
                solver_candidate_score, solver_guess_candidate,
                _CUDA_scratch_float, _CUDA_scratch_int);
        }
        else{
            // Compute the scores of each candidate word
            score_compute_main<<<static_cast<int>(num_words), __COMPUTE_BLOCKDIM__>>>(num_patterns);
            // Now take the argmax of all the scores
            cuda_max(_CUDA_word_scores, NULL, static_cast<int>(num_words), 
                solver_candidate_score, solver_guess_candidate,
                _CUDA_scratch_float, _CUDA_scratch_int);
        }
        cudaDeviceSynchronize();

        auto compute_end = timestamp;

        // Verbose Mode Function: Print the elected word
        int idx;
        float score;
        cudaMemcpy(&idx, solver_guess_candidate, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&score, solver_candidate_score, sizeof(float), cudaMemcpyDeviceToHost);
        printf("Candidate Elected [%d]: ", idx);
        word_print(_CPU_wordlist[idx], 0, ' ');
        printf("Candidate Score: %f\n", score);

        // Launch device kernel to obtain feedback:
        get_candidate_feedback<<<1,32>>>(test_idx);
        cudaDeviceSynchronize();
        // Verbose Mode Function: Print out the Feedback:
        coloring_t feedback;
        cudaMemcpy(&feedback, solver_feedback, sizeof(coloring_t), cudaMemcpyDeviceToHost);
        printf("Feedback: ");
        word_print(_CPU_wordlist[idx], feedback, ' ');
        printf("(Answer : ");
        word_print(_CPU_wordlist[test_idx], 0, ')');
        printf("\n");

        auto update_start = timestamp;

        // After feedback is obtained, launch kernel to perform update to the prior list
        int num_blocks = ceil_xdivy_int(static_cast<int>(num_words), BLOCKDIM);
        update_prior<<<num_blocks, BLOCKDIM>>>();   
        // Now tally up the number of remaining words as well as the prior sum
        cuda_reduce_sum_count_nonzeros(_CUDA_priorlist, num_words, 
            solver_prior_sum, solver_words_remaining, 
            _CUDA_scratch_float, _CUDA_scratch_int);
        cudaDeviceSynchronize();

        // Verbose Mode Function: Print out the valid words:
        std::cout << "Number of Words Remaining: ";
        int num_remaining;
        std::vector<float> prior_report(num_words);
        cudaMemcpy(&num_remaining, solver_words_remaining, sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << num_remaining << ":\n";
        cudaMemcpy(&(prior_report[0]), _CUDA_priorlist, sizeof(float) * num_words, cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_words; i++){
            if(prior_report[i] > 1e-12) word_print(_CPU_wordlist[i], 0, ' ');
        }
        std::cout << "\n" << std::flush;

        auto update_end = timestamp;

        // After each round, the GPU needs to communicate the current round's
        // feedback as well as the number of words remaining to the CPU
        cudaMemcpy(&words_remaining, solver_words_remaining, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&current_feedback, solver_feedback, sizeof(coloring_t), cudaMemcpyDeviceToHost);

        auto sequence_end = timestamp;
        

        // Verbose Mode: Report time tables
        std::cout << "=====================================================\n";
        std::cout << "Compute Time: " << TIME(compute_start, compute_end) << "\n";
        std::cout << "Feedback Time:" << TIME(compute_end, update_start) << "\n";
        std::cout << "Update Time: " << TIME(update_start, update_end) << "\n";
        std::cout << "Communicate Time: " << TIME(update_end, sequence_end) << "\n";
        std::cout << "\n\n";

        iterations += 1;
        if(is_correct_guess(current_feedback)) return iterations;
    }
    return iterations;
}


int CudaSolver::solve(int test_idx){
    // Initialize the solver temporary variables
    cudaMemcpy(_CUDA_priorlist, _CPU_priorlist, sizeof(float) * num_words, cudaMemcpyHostToDevice);
    cudaMemcpy(solver_prior_sum, &prior_sum, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(solver_words_remaining, &num_words, sizeof(size_t), cudaMemcpyHostToDevice);
    // Keep track of the number of words remaining on the CPU side as well.
    int words_remaining = static_cast<int>(num_words);
    // Obtain the actual number of patterns to be passed into the kernel
    int num_patterns = get_num_patterns(); 
    int iterations = 0;
    coloring_t current_feedback;
    while(iterations < MAXITERS){ // 10 iterations max to guarantee correctness
        if(words_remaining <= 2){
            // If the number of valid words remaining is at most 2, 
            // just perform random guess by taking the argmax
            cuda_max(_CUDA_priorlist, NULL, static_cast<int>(num_words), 
                solver_candidate_score, solver_guess_candidate,
                _CUDA_scratch_float, _CUDA_scratch_int);
        }
        else{
            // Compute the scores of each candidate word
            score_compute_main<<<static_cast<int>(num_words), __COMPUTE_BLOCKDIM__>>>(num_patterns);
            // Now take the argmax of all the scores
            cuda_max(_CUDA_word_scores, NULL, static_cast<int>(num_words), 
                solver_candidate_score, solver_guess_candidate,
                _CUDA_scratch_float, _CUDA_scratch_int);
        }
        // Launch device kernel to obtain feedback:
        get_candidate_feedback<<<1,32>>>(test_idx);
        // After feedback is obtained, launch kernel to perform update to the prior list
        int num_blocks = ceil_xdivy_int(static_cast<int>(num_words), BLOCKDIM);
        update_prior<<<num_blocks, BLOCKDIM>>>();   
        // Now tally up the number of remaining words as well as the prior sum
        cuda_reduce_sum_count_nonzeros(_CUDA_priorlist, num_words, 
            solver_prior_sum, solver_words_remaining, 
            _CUDA_scratch_float, _CUDA_scratch_int);

        // After each round, the GPU needs to communicate the current round's
        // feedback as well as the number of words remaining to the CPU
        cudaMemcpy(&words_remaining, solver_words_remaining, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&current_feedback, solver_feedback, sizeof(coloring_t), cudaMemcpyDeviceToHost);
        
        iterations += 1;
        if(is_correct_guess(current_feedback)) return iterations;
    }
    return iterations;
}