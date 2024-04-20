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

#include <omp.h>

// Global Parameter: Maximum word length in use
int wordlen = MAXLEN;

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()

unsigned long ceil_xdivy(unsigned long X, unsigned long Y){
    return (X + (Y - 1)) / Y;
}

void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << "-f <word list> -n <thread count> [-p <prior weights> -t <test list> -m <maximum word length> -r -v] \n";
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

/**
 * Verbose Mode Solver: Requires word list for information output.
 * @param prior - The prior weights of each word. 
 * @param pattern_matrix the coloring pattern matrrx.
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @warning This function destructively modifies the priors vector.
*/
void solver_verbose(wordlist_t &words,
            priors_t &priors,
            std::vector<std::vector<coloring_t>> &pattern_matrix,
            int &answer,
            float prior_sum){
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


    for(int k = 0; k < 10; k ++){
        /******************** Entropy Computation Phase **********************/
        std::cout<<"==========================================================\n";
        random_select = false;
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(priors);
            random_select = true;
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            auto compute_start = timestamp;
            #pragma omp parallel for schedule(dynamic) private(probability_scratch)
            for(int word_idx = 0; word_idx < num_words; word_idx++){
                probability_scratch.assign(num_patterns, 0.0f);
                // Pool up the total word weights for each pattern
                scatter_reduce(pattern_matrix[word_idx], priors,
                    probability_scratch);
                // Normalize the pooled weights into a probability distribution
                // Compute the entropy via map reduce
                entropys[word_idx] = entropy_compute(probability_scratch, 
                    prior_sum)+ (priors[word_idx] / prior_sum);
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

        auto update_end = timestamp;
        std::cout << "Update Phase total Time:" << TIME(update_start, update_end) << "\n"; 

        std::cout << "Proposed Guess: ";
        word_print(words[guess], feedback);
        if(!random_select)
        std::cout << "Expected Entropy: " << entropys[guess] << "\n";

        std::cout << "Remaining Uncertainty: " << uncertainty << "\n";
        std::cout << "Remaining Words (" << words_remaining <<"):\n";
        for(int i = 0; i < num_words; i++){
            if(!is_zero(priors[i])) word_print(words[i], 0, ' ');
        }
        std::cout << "\n";
        if(is_correct_guess(feedback)) break;
    }
}



/**
 * The main solver routine. Eliminated the need to input the word list.
 * The final word answer is coded as a word index in the word list.
 * @param prior - The prior weights of each word. 
 * @param pattern_matrix the coloring pattern matrrx.
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @warning This function destructively modifies the priors vector.
*/
void solver(priors_t &priors,
            std::vector<std::vector<coloring_t>> &pattern_matrix,
            int &answer,
            float prior_sum){
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

    for(int k = 0; k < 10; k ++){
        /******************** Entropy Computation Phase **********************/
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(priors);
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            #pragma omp parallel for schedule(dynamic) private(probability_scratch)
            for(int word_idx = 0; word_idx < num_words; word_idx++){
                probability_scratch.assign(num_patterns, 0.0f);
                // Pool up the total word weights for each pattern
                scatter_reduce(pattern_matrix[word_idx], priors,
                    probability_scratch);
                // Normalize the pooled weights into a probability distribution
                // Compute the entropy via map reduce, add the prior probability
                entropys[word_idx] = entropy_compute(probability_scratch, 
                    prior_sum) + (priors[word_idx] / prior_sum);
            }
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
        }

        // Check for guess feed back.
        feedback = pattern_matrix[guess][answer];
        if(is_correct_guess(feedback)) break;
        /******************** Update Phase **********************/
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
        // Compute the new uncertainty measure after a guess
        // uncertainty = entropy_compute(priors, prior_sum);
    }
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
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "f:n:p:t:m:rv")) != -1) {
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
        std::cout << "Initialization: " << TIME(init_start, init_end) << "\n";

    auto precompute_start = timestamp;
    // Precompute the coloring matrix
    compute_patterns(pattern_matrix, words);
    auto precompute_end = timestamp;

    std::cout << "Pre-processing: " << TIME(precompute_start, precompute_end) << "\n";
    // Benchmark all words in the test set.
    double time_total;
    int answer_index;
    priors_t prior_compute(priors.size()); // Makes a deep copy for each benchmark

    auto answer_start = timestamp;
    for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
        std::copy(priors.begin(), priors.end(), prior_compute.begin());
        answer_index = test_set[i];
        if(verbose){
            std::cout << "Benchmarking word: ";
            word_print(words[answer_index]);
            solver_verbose(words, prior_compute, pattern_matrix, answer_index, priors_sum);
        }
        else{
            solver(prior_compute, pattern_matrix, answer_index, priors_sum);
        }
    }

    auto answer_end = timestamp;
    time_total = TIME(answer_start, answer_end);

    double average_time = time_total / static_cast<double>(test_set.size());
    std::cout << "Average time taken: " << average_time << " sec per word\n";


    return 0;
}
