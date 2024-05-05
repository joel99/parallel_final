#include "word.h"
#include "utils.h"
#include "mathutils.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <list>
#include <omp.h>

#define MAXITERS 10

// Global Parameter: Maximum word length in use
int wordlen = MAXLEN;

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()


void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << " -f <word list> -n <thread count> [-p <prior weights> -t <test list> -m <maximum word length> -x <parallel mode> -r -v] \n";
    std::cout << "-v: verbose mode\n-r: use randomized priors";
    std::cout << "-m: specifies the maximum word length. Must be in between 1 and 8 (default)";
    std::cout << "-x: specifies the parallelization strategy. 0 for guess parallel, 1 for candidate parallel";
    std::cout << "The test list must contain words in the word list\n";
    return;
}

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T, typename A>
int arg_min(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), min_element(vec.begin(), vec.end())));
}


/**
 * This function computes the entirety of the coloring pattern matrix
 * using pair-wise word comparison. This is a massive data parallel procedure
 * @param pattern_matrix The pattern matrix to be written to
 * @param words The list of words
*/
void compute_patterns(std::vector<std::vector<coloring_t>> &pattern_matrix,
                      wordlist_t &words){
    int num_words = words.size();
    #pragma omp parallel for schedule(dynamic)
    for(int query_idx = 0; query_idx < num_words; query_idx++){
        word_t query = words[query_idx];
        #pragma omp simd
        for (int candidate_idx = 0; candidate_idx < num_words; candidate_idx++){
            pattern_matrix[query_idx][candidate_idx] = 
                word_cmp(query, words[candidate_idx]);
        }
    }
}

/**
 * Verbose Mode Solver: Requires word list for information output.
 * @param prior - The prior weights of each word. 
 * @param pattern_matrix the coloring pattern matrrx.
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @param mode - 'g' for guess parallel, 'c' for candidate parallel, 'h' for hybrid
 * @param capacity - The capacity of the scratch space (hybrid)
 * @param rebuild - Whether to rebuild words across iterations
 * @param scatter_scratch - Scratch space for scatter reduce (hybrid), shape (num_threads, num_words, num_patterns)
 * @param locks - Locks for each guess (hybrid)
 * @warning This function destructively modifies the priors vector.
*/
int_fast64_t solver_verbose(wordlist_t &words,
            priors_t &priors,
            std::vector<std::vector<coloring_t>> &pattern_matrix,
            std::vector<index_t> &src_idx,
            std::vector<index_t> &src_idx_scratch,
            priors_t &priors_scratch,
            std::vector<std::vector<coloring_t>> &patterns_scratch,
            int &answer,
            float prior_sum,
            char mode,
            int capacity,
            bool rebuild,
            std::vector<std::vector<std::vector<float>>> &scatter_scratch,
            std::vector<omp_lock_t> &locks){
    auto in_start = timestamp;
    // Initialize Additional Solver Data
    int num_words = pattern_matrix.size();
    int words_remaining = num_words;
    int num_patterns = get_num_patterns();
    // Scratch work and entrypy storage.
    std::vector<float> probability_scratch;
    std::vector<float> entropys(num_words, 0.0f);

    int guess; // The index to the guessed word
    coloring_t feedback;
    // Computes the initial uncertainty measure
    float uncertainty = entropy_compute(priors, prior_sum);

    std::cout << "Initial Uncertainty: " << uncertainty << "\n";
    bool random_select;

    int iters = 0;

    // For the first round, just use original data pointers (don't copy, huge overhead!)
    std::vector<index_t>* src_idx_ref = &src_idx;  
    priors_t* priors_ref = &priors; 
    std::vector<std::vector<coloring_t>>* patterns_ref = &pattern_matrix; 
    if (rebuild) { // Resize
        src_idx_scratch.resize(num_words);
        priors_scratch.resize(num_words);
        patterns_scratch.resize(num_words);
        for (int i = 0; i < num_words; i++){
            patterns_scratch[i].resize(pattern_matrix[i].size());
        }
    }

    auto loop_start = timestamp;
    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        std::cout<<"==========================================================\n";
        random_select = false;
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            if (!rebuild) {
                guess = arg_max(priors);
            } else {
                guess = (*src_idx_ref)[0];
            }
            random_select = true;
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            auto compute_start = timestamp;
            if (mode == 's') {
                for(int word_idx = 0; word_idx < num_words; word_idx++){
                    probability_scratch.assign(num_patterns, 0.0f);
                    scatter_reduce((*patterns_ref)[word_idx], *priors_ref,
                        probability_scratch);
                    entropys[word_idx] = entropy_compute(probability_scratch, 
                        prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                }
            } else if (mode == 'g') {
                #pragma omp parallel for schedule(dynamic) private(probability_scratch)
                for(int word_idx = 0; word_idx < num_words; word_idx++){
                    probability_scratch.assign(num_patterns, 0.0f);
                    // Pool up the total word weights for each pattern
                    // scatter_reduce(pattern_matrix[word_idx], priors,
                    scatter_reduce((*patterns_ref)[word_idx], *priors_ref,
                        probability_scratch);
                    // Normalize the pooled weights into a probability distribution
                    // Compute the entropy via map reduce, add the prior probability
                    entropys[word_idx] = entropy_compute(probability_scratch, 
                        // prior_sum) + (priors_scratch[word_idx] / prior_sum);
                        prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                }
            } else if (mode == 'c') { // Candidate parallel - absurdly slow
                if (rebuild) {
                    exit(1);
                } else {
                    // candidate_parallel from sred.cpp - far too slow due to repeated overhead
                    // probability_scratch.resize(num_patterns);  // Resize before the parallel block
                    // float out = 0.0f;
                    // #pragma omp parallel
                    // {
                    //     for(int word_idx = 0; word_idx < num_words; word_idx++){
                    //         #pragma omp for
                    //         for (int i = 0; i < num_patterns; i++){
                    //             probability_scratch[i] = 0.0f;
                    //         }

                    //         parallel_scatter_reduce(pattern_matrix[word_idx], priors,
                    //             probability_scratch);
                            
                    //         parallel_entropy_compute(probability_scratch, prior_sum, out);

                    //         #pragma omp single nowait
                    //         entropys[word_idx] = out + (*priors_ref)[word_idx] / prior_sum;
                    //     }
                    // }

                    int candidates = static_cast<int>((*patterns_ref)[0].size());
                    std::vector<std::vector<std::vector<float>>>
                        scratch(omp_get_max_threads(), std::vector<std::vector<float>>(num_words, std::vector<float>(num_patterns, 0.0f)));
                    std::vector<std::vector<float>> data_out(num_words, std::vector<float>(num_patterns, 0.0f));
                    // std::cout << "Candidates: " << candidates << "\n";
                    // std::cout << "Colors: " << num_patterns << "\n";
                    // scratch of size thread x input x num_patterns
                    int candidate_span = ceil_xdivy(candidates, omp_get_num_threads());
                    auto inner_start = timestamp;
                    #pragma omp parallel
                    {
                        int thread_id = omp_get_thread_num();
                        int read_min = candidate_span * thread_id;
                        int read_max = std::min(read_min + candidate_span, candidates);
                        int idx;
                        for (int guess = 0; guess < num_words; guess++){
                            for(int candidate = read_min; candidate < read_max; candidate++){
                                idx = (*patterns_ref)[guess][candidate];
                                scratch[thread_id][guess][idx] += (*priors_ref)[candidate];
                            }
                            #pragma omp critical
                            {
                                for(int color = 0; color < num_patterns; color++){
                                    data_out[guess][color] += scratch[thread_id][guess][color];
                                }
                            }
                        }
                        #pragma omp barrier
                        #pragma omp for
                        for (int word_idx = 0; word_idx < num_words; word_idx++){
                            {
                            entropys[word_idx] = entropy_compute(data_out[word_idx], prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                            }
                        }
                    }
                    auto inner_end = timestamp;
                    std::cout << "Inner Time: " << TIME(inner_start, inner_end) << "\n";
                }
            } else if (mode == 'h') {
                int num_threads = omp_get_max_threads();
                scatter_scratch.assign(num_threads, std::vector<std::vector<float>>(capacity, std::vector<float>(num_patterns, 0.0f)));
                std::vector<std::vector<float>> data_out(num_words, std::vector<float>(num_patterns, 0.0f));
                auto inner_start = timestamp;
                scatter_reduce_cap((*priors_ref), (*patterns_ref), data_out, scatter_scratch, locks);
                auto scatter_end = timestamp;
                auto entropy_start = timestamp;
                #pragma omp parallel for
                for (int word_idx = 0; word_idx < num_words; word_idx++){
                    {
                    entropys[word_idx] = entropy_compute(data_out[word_idx], prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                    }
                }
                auto inner_end = timestamp;
                // std::cout << "Inner Time: " << TIME(inner_start, inner_end) << "\n";
                // std::cout << "Scatter Time: " << TIME(inner_start, scatter_end) << "\n";
                // std::cout << "Entropy Time: " << TIME(entropy_start, inner_end) << "\n";
            }
            auto compute_end = timestamp;
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            auto select_end = timestamp;
            std::cout << "Scatter Reduce + Entropy Computation Time: " << TIME(compute_start, compute_end) << "\n";
            std::cout << "Word Selection Time:" << TIME(compute_end, select_end) << "\n";
        }
        
        // Check for guess feed back.
        feedback = pattern_matrix[guess][answer];

        /******************** Update Phase **********************/
        auto update_start = timestamp;
        if (!rebuild) {
            words_remaining = 0;
            prior_sum = 0.0f;
            for(int i = 0; i < num_words; i++){
                if(is_zero(priors[i])) continue; // prior == 0 for invalid
                if(pattern_matrix[guess][i] != feedback) priors[i] = 0.0f;
                else{
                    words_remaining += 1;
                    prior_sum += priors[i];
                }
            }
            // Compute the new uncertainty measure after a guess
            uncertainty = entropy_compute(priors, prior_sum);
        } else {
            // guess and entropy are in original space, others are reduced
            prior_sum = 0.0f;
            int _write = 0;
            for (int _read = 0; _read < words_remaining; _read++){
                if ((*patterns_ref)[guess][_read] == feedback) {
                    int prior_read = (*priors_ref)[_read];
                    priors_scratch[_write] = prior_read;
                    prior_sum += prior_read;
                    src_idx_scratch[_write] = (*src_idx_ref)[_read];
                    for (int k = 0; k < num_words; k++){
                        patterns_scratch[k][_write] = (*patterns_ref)[k][_read];
                    }
                    _write++;
                }
            }
            words_remaining = _write;
            src_idx_scratch.resize(words_remaining);
            priors_scratch.resize(words_remaining);
            for (int i = 0; i < num_words; i++){
                patterns_scratch[i].resize(words_remaining);
            }
            src_idx_ref = &src_idx_scratch;
            priors_ref = &priors_scratch;
            patterns_ref = &patterns_scratch;

            uncertainty = entropy_compute(priors_scratch, prior_sum);
        }


        auto update_end = timestamp;
        std::cout << "Update Phase total Time:" << TIME(update_start, update_end) << "\n"; 

        std::cout << "Proposed Guess: ";
        word_print(words[guess], feedback);
        if(!random_select)
        std::cout << "Expected Entropy: " << entropys[guess] << "\n";

        std::cout << "Remaining Uncertainty: " << uncertainty << "\n";
        std::cout << "Remaining Words (" << words_remaining <<"):\n";
        if (!rebuild) {
            for(int i = 0; i < num_words; i++){
                if(!is_zero(priors[i])) word_print(words[i], 0, ' ');
            }
        } else {
            for(int i = 0; i < words_remaining; i++){
                word_print(words[(*src_idx_ref)[i]], 0, ' ');
            }
        }
        std::cout << "\n";

        iters ++;
        if(is_correct_guess(feedback)) {
            auto in_end = timestamp;
            std::cout << "Loop Time: " << TIME(loop_start, in_end) << "\n";
            std::cout << "Total Time: " << TIME(in_start, in_end) << "\n";
            return iters;
        }
    }
    auto in_end = timestamp;
    std::cout << "Loop Time: " << TIME(loop_start, in_end) << "\n";
    std::cout << "Total Time: " << TIME(in_start, in_end) << "\n";
    return iters;
}



/**
 * The main solver routine. Eliminated the need to input the word list.
 * The final word answer is coded as a word index in the word list.
 * @param prior - The prior weights of each word. 
 * @param pattern_matrix the coloring pattern matrrx.
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param mode - 'g' for guess parallel, 'c' for candidate parallel, 'h' for hybrid
 * @param capacity - The capacity of the scratch space (hybrid )
 * @param answer - The WORD INDEX of the correct word.
 * @param capacity - The capacity of the scratch space (hybrid)
 * @param rebuild - Whether to rebuild words across iterations
 * @param scatter_scratch - Scratch space for scatter reduce (hybrid), shape (num_threads, num_words, num_patterns)
 * @param locks - Locks for each guess (hybrid)
 * @warning This function destructively modifies the priors vector.
*/
int solver(priors_t &priors,
            std::vector<std::vector<coloring_t>> &pattern_matrix,
            std::vector<index_t> &src_idx,
            std::vector<index_t> &src_idx_scratch,
            priors_t &priors_scratch,
            std::vector<std::vector<coloring_t>> &patterns_scratch,
            int &answer,
            float prior_sum,
            char mode,
            int capacity,
            bool rebuild,
            std::vector<std::vector<std::vector<float>>> &scatter_scratch,
            std::vector<omp_lock_t> &locks){
    // Initialize Additional Solver Data
    int num_words = pattern_matrix.size();
    int words_remaining = num_words;
    int num_patterns = get_num_patterns();
    // Scratch work and entrypy storage.
    std::vector<float> probability_scratch;
    std::vector<float> entropys(num_words, 0.0f);

    int guess; // The index to the guessed word
    coloring_t feedback;
    // Computes the initial uncertainty measure
    // float uncertainty = entropy_compute(priors, prior_sum);

    int iters = 0;

    std::vector<index_t>* src_idx_ref = &src_idx;  
    priors_t* priors_ref = &priors; 
    std::vector<std::vector<coloring_t>>* patterns_ref = &pattern_matrix; 
    if (rebuild) { // Resize
        src_idx_scratch.resize(num_words);
        priors_scratch.resize(num_words);
        patterns_scratch.resize(num_words);
        for (int i = 0; i < num_words; i++){
            patterns_scratch[i].resize(pattern_matrix[i].size());
        }
    }
    
    std::vector<float> inner_time = std::vector<float>(MAXITERS, 0.0f);
    std::vector<float> scatter_time = std::vector<float>(MAXITERS, 0.0f);
    std::vector<float> entropy_time = std::vector<float>(MAXITERS, 0.0f);
    std::vector<float> rebuild_time = std::vector<float>(MAXITERS, 0.0f);
    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            if (!rebuild) {
                guess = arg_max(priors);
            } else {
                guess = (*src_idx_ref)[0];
            }
        } else {
            if (mode == 's') { // serial
                auto inner_start = timestamp;
                for(int word_idx = 0; word_idx < num_words; word_idx++){
                    probability_scratch.assign(num_patterns, 0.0f);
                    auto scatter_start = timestamp;
                    scatter_reduce((*patterns_ref)[word_idx], *priors_ref,
                        probability_scratch);
                    auto scatter_end = timestamp;
                    entropys[word_idx] = entropy_compute(probability_scratch, 
                        prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                    auto entropy_end = timestamp;
                    scatter_time[iters] += TIME(scatter_start, scatter_end);
                    entropy_time[iters] += TIME(scatter_end, entropy_end);
                }
                auto inner_end = timestamp;
                inner_time[iters] = TIME(inner_start, inner_end);
            } else if (mode == 'g') { // More than 2 words: Compute the entropy for ALL words
                auto inner_start = timestamp;
                #pragma omp parallel for schedule(dynamic) private(probability_scratch)
                for(int word_idx = 0; word_idx < num_words; word_idx++){
                    probability_scratch.assign(num_patterns, 0.0f);
                    // Pool up the total word weights for each pattern
                    // scatter_reduce(pattern_matrix[word_idx], priors,
                    scatter_reduce((*patterns_ref)[word_idx], *priors_ref,
                        probability_scratch);
                    // Normalize the pooled weights into a probability distribution
                    // Compute the entropy via map reduce, add the prior probability
                    entropys[word_idx] = entropy_compute(probability_scratch, 
                        // prior_sum) + (priors_scratch[word_idx] / prior_sum);
                        prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                }
                auto inner_end = timestamp;
                inner_time[iters] = TIME(inner_start, inner_end);
            } else if (mode == 'c') { // Candidate parallel - absurdly slow
                if (rebuild) {
                    // not implemented - rebuilding gains comes from reducing scatter read, not scatter write
                    // candidate parallel bottlenecked by write
                    exit(1);
                } else {
                    // False starts in integration - needed to minimize single/critical paths 
                    // Entropy computation moved to separate loop
                    // use extra memory for scratch to avoid in-loop assignment
                    // reduction_scatter_reduce from sred_2d.cpp (manual)
                    int candidates = static_cast<int>((*patterns_ref)[0].size());
                    // work to refactor scratch outside of candidate parallel not done - mainly because candidate parallel is not promising
                    std::vector<std::vector<std::vector<float>>>
                        scratch(omp_get_max_threads(), std::vector<std::vector<float>>(num_words, std::vector<float>(num_patterns, 0.0f)));
                    std::vector<std::vector<float>> data_out(num_words, std::vector<float>(num_patterns, 0.0f));
                    // std::cout << "Colors: " << num_patterns << "\n";
                    // scratch of size thread x input x num_patterns
                    int candidate_span = ceil_xdivy(candidates, omp_get_num_threads());
                    auto inner_start = timestamp;
                    auto scatter_start = timestamp;
                    
                    #pragma omp parallel
                    {
                        int thread_id = omp_get_thread_num();
                        int read_min = candidate_span * thread_id;
                        int read_max = std::min(read_min + candidate_span, candidates);
                        int idx;
                        #pragma omp single
                        scatter_start = timestamp;
                        for (int guess = 0; guess < num_words; guess++){
                            for(int candidate = read_min; candidate < read_max; candidate++){
                                idx = (*patterns_ref)[guess][candidate];
                                scratch[thread_id][guess][idx] += (*priors_ref)[candidate];
                            }
                            #pragma omp critical
                            {
                                for(int color = 0; color < num_patterns; color++){
                                    data_out[guess][color] += scratch[thread_id][guess][color];
                                }
                            }
                        }
                        // profiling indicates read are order of magnitude faster than write i.e. first thread reaches end of scatter in appropriately sped up time.
                        // unclear why we're not seeing same speedup as in sred.cpp
                    }
                    auto scatter_end = timestamp;
                    auto scatter_time = TIME(scatter_start, scatter_end);
                    // separate scatter / entropy for refactorability, ease of reading, benchmarking
                    auto entropy_start = timestamp;
                    #pragma omp parallel for schedule(dynamic)
                    for (int word_idx = 0; word_idx < num_words; word_idx++){
                        {
                        entropys[word_idx] = entropy_compute(data_out[word_idx], prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                        }
                    }
                    auto entropy_end = timestamp;
                    auto entropy_time = TIME(entropy_start, entropy_end);
                    auto inner_end = timestamp;
                    // std::cout << "Inner Time: " << TIME(inner_start, inner_end) << "\n";
                    // std::cout << "Scatter Time: " << scatter_time << "\n";
                    // std::cout << "Entropy Time: " << entropy_time << "\n";
                }
            } else if (mode == 'h') {
                // hybrid strategy - scatter_reduce_cap from sred_2d
                int num_threads = omp_get_max_threads();
                scatter_scratch.assign(num_threads, std::vector<std::vector<float>>(capacity, std::vector<float>(num_patterns, 0.0f)));
                std::vector<std::vector<float>> data_out(num_words, std::vector<float>(num_patterns, 0.0f));
                scatter_reduce_cap((*priors_ref), (*patterns_ref), data_out, scatter_scratch, locks);

                #pragma omp parallel for
                for (int word_idx = 0; word_idx < num_words; word_idx++){
                    {
                    entropys[word_idx] = entropy_compute(data_out[word_idx], prior_sum) + ((*priors_ref)[word_idx] / prior_sum);
                    }
                }
            }
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
        }

        // Check for guess feed back.
        feedback = pattern_matrix[guess][answer];
        if(is_correct_guess(feedback)) {
            // sum nonnegative iters
            float inner_time_total = 0.0f;
            float scatter_time_total = 0.0f;
            float entropy_time_total = 0.0f;
            float rebuild_time_total = 0.0f;
            for (int i = 0; i < iters; i++) {
                inner_time_total += inner_time[i];
                scatter_time_total += scatter_time[i];
                entropy_time_total += entropy_time[i];
                rebuild_time_total += rebuild_time[i];
            }
            std::cout << "\nInner Time: " << inner_time_total << "\n";
            std::cout << "Scatter Time: " << scatter_time_total << "\n";
            std::cout << "Entropy Time: " << entropy_time_total << "\n";
            std::cout << "Rebuild Time: " << rebuild_time_total << "\n";
            for (int i = 0; i < iters; i++) {
                std::cout << "Iteration " << i << " Round Time: " << inner_time[i] << "\n";
            }
            return iters + 1;
        }
        /******************** Update Phase **********************/
        auto update_start = timestamp;
        if (!rebuild) {
            words_remaining = 0;
            prior_sum = 0.0f;
            #pragma omp parallel for schedule(dynamic, 64) reduction(+:words_remaining) reduction(+:prior_sum)
            for(int i = 0; i < num_words; i++){
                if(is_zero(priors[i])) continue; // prior == 0 for invalid
                if(pattern_matrix[guess][i] != feedback) priors[i] = 0.0f;
                else{
                    words_remaining += 1;
                    prior_sum += priors[i];
                }
            }
        } else {
            // Rebuild by pushing values to start of arrays
            // First deep copy
            prior_sum = 0.0f;

            int _write = 0;  // This will keep track of the global write index after parallel computation

            // concat reduce parallel
            // #pragma omp parallel
            // {
            //     int local_write = 0;
            //     std::vector<int> local_priors_scratch;
            //     std::vector<int> local_src_idx_scratch;
            //     std::vector<std::vector<int>> local_patterns_scratch(num_words);

            //     // Reduction variables for local thread
            //     int local_prior_sum = 0;

            //     #pragma omp for schedule(dynamic, 64)
            //     for (int _read = 0; _read < words_remaining; _read++) {
            //         if ((*patterns_ref)[guess][_read] == feedback) {
            //             int prior_read = (*priors_ref)[_read];
            //             local_priors_scratch.push_back(prior_read);
            //             local_prior_sum += prior_read;
            //             local_src_idx_scratch.push_back((*src_idx_ref)[_read]);

            //             for (int k = 0; k < num_words; k++) {
            //                 local_patterns_scratch[k].push_back((*patterns_ref)[k][_read]);
            //             }
            //             local_write++;
            //         }
            //     }

            //     // Critical section to merge local buffers into the global buffers
            //     #pragma omp critical
            //     {
            //         int start_idx = _write;
            //         for (int i = 0; i < local_write; i++) {
            //             priors_scratch[start_idx + i] = local_priors_scratch[i];
            //             src_idx_scratch[start_idx + i] = local_src_idx_scratch[i];
            //             for (int k = 0; k < num_words; k++) {
            //                 patterns_scratch[k][start_idx + i] = local_patterns_scratch[k][i];
            //             }
            //         }
            //         _write += local_write;  // Update the global index after local writes
            //         prior_sum += local_prior_sum;  // Accumulate the sum of priors
            //     }
            // }

            // serial rebuild
            for (int _read = 0; _read < words_remaining; _read++){
                if ((*patterns_ref)[guess][_read] == feedback) {
                    int prior_read = (*priors_ref)[_read];
                    priors_scratch[_write] = prior_read;
                    prior_sum += prior_read;
                    src_idx_scratch[_write] = (*src_idx_ref)[_read];
                    for (int k = 0; k < num_words; k++){
                        patterns_scratch[k][_write] = (*patterns_ref)[k][_read];
                    }
                    _write++;
                }
            }
            words_remaining = _write;
            src_idx_scratch.resize(words_remaining);
            priors_scratch.resize(words_remaining);
            for (int i = 0; i < num_words; i++){
                patterns_scratch[i].resize(words_remaining);
            }
            src_idx_ref = &src_idx_scratch;
            priors_ref = &priors_scratch;
            patterns_ref = &patterns_scratch;
        }
        auto update_end = timestamp;
        rebuild_time[iters] = TIME(update_start, update_end);
        iters ++;
    }
    float inner_time_total = 0.0f;
    float scatter_time_total = 0.0f;
    float entropy_time_total = 0.0f;
    float rebuild_time_total = 0.0f;
    for (int i = 0; i < iters; i++) {
        inner_time_total += inner_time[i];
        scatter_time_total += scatter_time[i];
        entropy_time_total += entropy_time[i];
        rebuild_time_total += rebuild_time[i];
    }
    std::cout << "\nInner Time: " << inner_time_total << "\n";
    std::cout << "Scatter Time: " << scatter_time_total << "\n";
    std::cout << "Entropy Time: " << entropy_time_total << "\n";
    std::cout << "Rebuild Time: " << rebuild_time_total << "\n";
    for (int i = 0; i < iters; i++) {
        std::cout << "Iteration " << i << " Round Time: " << inner_time[i] << "\n";
    }
    return iters;
}

int main(int argc, char **argv) {
    auto init_start = timestamp;


    // Initialization Stage
    std::string text_filename;
    std::string test_filename;
    std::string prior_filename;
    int num_threads = 0;
    bool verbose = false;
    bool rand_prior = false;
    bool rebuild = false;
    int opt;
    int capacity = 4;
    char mode = '\0';
    // Read program parameters
    while ((opt = getopt(argc, argv, "f:n:p:t:m:c:x:brv")) != -1) {
        switch (opt) {
        case 'f':
            text_filename = optarg;
            break;
        case 't':
            test_filename = optarg;
            break;
        case 'p':
            prior_filename = optarg;
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        case 'm':
            wordlen = atoi(optarg);
            break;
        case 'c':
            capacity = atoi(optarg);
            break;
        case 'x':
            mode = *optarg;
            break;
        case 'b':
            rebuild = true;
            break;
        case 'r':
            rand_prior = true;
            break;
        case 'v':
            verbose = true;
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }
    if(empty(text_filename) || num_threads <= 0){
        usage(argv[0]);
        exit(1);
    }
    if(wordlen <= 0 || wordlen > MAXLEN){
        std::cerr << "Invalid Wordlen Parameter [" << wordlen << "]\n";
        exit(1);
    }
    
    omp_set_num_threads(num_threads);

    // Initializing word list
    wordlist_t words;
    if(read_words_from_file(text_filename, words)) exit(1);
    // Initialize prior weights
    priors_t priors;
    float priors_sum; // The sum of all prior weights.
    if(empty(prior_filename))
        if(rand_prior){
            priors = generate_random_priors(words.size(), priors_sum);
        } 
        else{
            priors = generate_uniform_priors(words.size(), priors_sum);
        }
    else{
        if(read_priors_from_file(prior_filename, priors_sum, priors)) exit(1);
        if(priors.size() != words.size()){  // Check if size match:
            std::cerr << "Input Files Length Mismatch!\n";
            exit(1);
        }
    }

    // Initialize test set: 
    std::vector<int> test_set;
    std::string linebuf;
    word_t buffer;
    if(!empty(test_filename)){ // Read test set from file
        if(read_test_set_from_file(test_filename, words, test_set)) exit(1);
    }
    else{
        test_set.resize(1);
        std::cout << "Test set is not provied. Please manually enter the answer word:\n";
        while(1){
            std::getline(std::cin, linebuf);
            if(linebuf.empty()) continue;
            str2word(linebuf, buffer);
            test_set[0] = list_query(words, buffer);
            if(test_set[0] < 0){
                std::cout << "The word '";
                word_print(buffer, 0, 0x20);
                std::cout << "' is not valid.\n";
            }
            else break;
        }
    }
  
    // Initialize all data structures used in the wordle solver
    std::vector<std::vector<coloring_t>> 
        pattern_matrix(words.size(), std::vector<coloring_t>(words.size()));

    // IO Complete
    auto init_end = timestamp;
        std::cout << "IO Initialization: " << TIME(init_start, init_end) << "\n";

    auto precompute_start = timestamp;
    // Precompute the coloring matrix
    compute_patterns(pattern_matrix, words);
    auto precompute_end = timestamp;

    std::vector<index_t> src_idx(words.size());
    int num_words = words.size();
    for (int i = 0; i < num_words; i++){
        src_idx[i] = i;
    }
    std::vector<index_t> src_idx_scratch = src_idx;
    priors_t priors_scratch = priors;
    std::vector<std::vector<coloring_t>> patterns_scratch = pattern_matrix;
    std::vector<std::vector<std::vector<float>>> scatter_scratch;
    std::vector<omp_lock_t> locks;
    if (mode == 'h') {
        scatter_scratch = std::vector<std::vector<std::vector<float>>>(num_threads, std::vector<std::vector<float>>(capacity, std::vector<float>(get_num_patterns(), 0.0f)));
        locks = std::vector<omp_lock_t>(num_words);
        for(int i = 0; i < num_words; i++){
            omp_init_lock(&locks[i]);
        }
    }
    std::cout << "Pre-processing: " << TIME(precompute_start, precompute_end) << "\n";
    if (rebuild) {
        std::cout << "Rebuild\n";
    } else {
        std::cout << "No Rebuild\n";
    }
    if (mode == 'c') {
        std::cout << "Parallel Mode: Candidate Parallel\n";
    } else if (mode == 'g') {
        std::cout << "Parallel Mode: Guess Parallel\n";
    } else if (mode == 's') {
        std::cout << "Parallel Mode: Serial\n";
    } else if (mode == 'h') {
        std::cout << "Parallel Mode: Hybrid\n";
    }
    // Benchmark all words in the test set.
    double time_total;
    int answer_index;
    priors_t prior_compute(priors.size()); // Makes a deep copy for each benchmark

    auto answer_start = timestamp;
    int rounds;
    for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
        std::copy(priors.begin(), priors.end(), prior_compute.begin());
        answer_index = test_set[i];
        std::cout << "Benchmarking word: ";
            word_print(words[answer_index], 0, ' ');
        
        if(verbose){
            rounds = solver_verbose(
                words, 
                prior_compute, 
                pattern_matrix, 
                src_idx, 
                src_idx_scratch,
                priors_scratch,
                patterns_scratch,
                answer_index, 
                priors_sum, 
                mode,
                capacity,
                rebuild,
                scatter_scratch,
                locks);
        }
        else{
            rounds = solver(
                prior_compute, 
                pattern_matrix, 
                src_idx, 
                src_idx_scratch,
                priors_scratch,
                patterns_scratch,
                answer_index,  
                priors_sum, 
                mode,
                capacity,
                rebuild,
                scatter_scratch,
                locks);
        }
        std::cout << "<" << rounds << ">\n";
    }

    auto answer_end = timestamp;
    time_total = TIME(answer_start, answer_end);

    double average_time = time_total / static_cast<double>(test_set.size());
    std::cout << "Average time taken: " << average_time << " sec per word\n";

    if (mode == 'h') {
        for(auto l:locks){
            omp_destroy_lock(&l);
        }
    }
    return 0;
}
