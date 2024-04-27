#include "word.h"
#include "utils.h"
#include "cuda_solver.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <unistd.h>
#include <stdlib.h>

// CUDA solver routines:
void CUDA_main(std::vector<word_t> &word_list, std::vector<float> &prior_list, std::vector<int> &test_list);

// Global Parameter: Maximum word length in use
int wordlen = MAXLEN;

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()

unsigned long ceil_xdivy(unsigned long X, unsigned long Y){
    return (X + (Y - 1)) / Y;
}

void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << " -f <word list> [-p <prior weights> -t <test list> -m <maximum word length> -r -v] \n";
    std::cout << "-v: verbose mode\n-r: use randomized priors";
    std::cout << "-m: specifies the maximum word length. Must be in between 1 and 8 (default)";
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

int main(int argc, char **argv) {
    auto init_start = timestamp;
    // Initialization Stage
    std::string text_filename;
    std::string test_filename;
    std::string prior_filename;
    bool verbose = false;
    bool rand_prior = false;
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "f:p:t:m:rv")) != -1) {
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
        case 'm':
            wordlen = atoi(optarg);
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
    if(empty(text_filename)){
        usage(argv[0]);
        exit(1);
    }
    if(wordlen <= 0 || wordlen > MAXLEN){
        std::cerr << "Invalid Wordlen Parameter [" << wordlen << "]\n";
        exit(1);
    }

    // CPU Code: Reading from file / Initializing the word list / prior list

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

    CudaSolver *Solver = new CudaSolver();
    Solver -> data_copy(words, priors, priors_sum);
    Solver -> cuda_setup();

    auto init_end = timestamp;
    std::cout << "Initialization: " << TIME(init_start, init_end) << "\n";

    auto precompute_start = timestamp;

    // Compute the pattern matrix
    Solver -> pattern_comute();

    auto precompute_end = timestamp;

    std::cout << "GPU Pre-processing: " << TIME(precompute_start, precompute_end) << "\n";

    std::vector<std::vector<coloring_t>> 
    pattern_matrix(words.size(), std::vector<coloring_t>(words.size()));

    // auto precompute_start1 = timestamp;
    // // Precompute the coloring matrix
    // compute_patterns(pattern_matrix, words);
    // auto precompute_end1 = timestamp;

    // std::cout << "CPU Pre-processing: " << TIME(precompute_start1, precompute_end1) << "\n";
    
    

    auto answer_start = timestamp;
    for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
        int answer_index = test_set[i];
        int iters = 0;
        std::cout << "Benchmarking word: ";
            word_print(words[answer_index], 0, ' ');
        if(verbose){
            iters = (Solver -> solve_verbose(answer_index));
        }
        else{
           iters = (Solver -> solve(answer_index));
        }
        std::cout << "<" << iters << ">\n";
    }
    auto answer_end = timestamp;
    double time_total = TIME(answer_start, answer_end);
    double average_time = time_total / static_cast<double>(test_set.size());
    std::cout << "Average time taken: " << average_time << " sec per word\n";






  

    // // IO Complete
    


    // 
    // // Benchmark all words in the test set.
    // double time_total;
    // int answer_index;
    // priors_t prior_compute(priors.size()); // Makes a deep copy for each benchmark

    // auto answer_start = timestamp;
    // for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
    //     std::copy(priors.begin(), priors.end(), prior_compute.begin());
    //     answer_index = test_set[i];
    //     if(verbose){
    //         std::cout << "Benchmarking word: ";
    //         word_print(words[answer_index]);
    //         solver_verbose(words, prior_compute, pattern_matrix, answer_index, priors_sum);
    //     }
    //     else{
    //         solver(prior_compute, pattern_matrix, answer_index, priors_sum);
    //     }
    // }

    // auto answer_end = timestamp;
    // time_total = TIME(answer_start, answer_end);

    // double average_time = time_total / static_cast<double>(test_set.size());
    // std::cout << "Average time taken: " << average_time << " sec per word\n";


    return 0;
}
