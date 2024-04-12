#include "word.h"
#include "utils.h"
#include "mathutils.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstring>
#include <unistd.h>

unsigned long ceil_xdivy(unsigned long X, unsigned long Y){
    return (X + (Y - 1)) / Y;
}

void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << "-f <word list> -n <thread count> [-p <prior probabilities>] \n";
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
    unsigned long num_words = words.size();
    for(int query_idx = 0; query_idx < num_words; query_idx++){
        word_t query = words[query_idx];
        for (int candidate_idx = 0; candidate_idx < num_words; candidate_idx++){
            pattern_matrix[query_idx][candidate_idx] = 
                word_cmp(query, words[candidate_idx]);
        }
    }
}

void solver_main(game_data_t &data,
                 word_t &answer,
                 wordlist_t &words,
                 priors_t &priors,
                 std::vector<std::vector<coloring_t>> pattern_matrix){
    // Initialize solver data
    int words_remaining = words.size();
        // The number of words consistent with the observed pattern
    int num_words = words.size();
        // The number of total candidate words
    int num_patterns = get_num_patterns();
    std::vector<bool> mask(words.size(), 1);
    std::vector<float> probability_scratch;
    std::vector<float> entropy_vector(num_words, 0.0f);

    int candidate_idx;
    coloring_t feedback;

    for(int k = 0; k < 10; k ++ ){
        std::cout << "Iteration " << data.size() << "\n";
        std::cout << "Words Remaining " << words_remaining << "\n";

        if(words_remaining <= 2){ // Edge Case: Random Guess
            candidate_idx = arg_max(mask);
        }
        else{
            for(int word_idx = 0; word_idx < num_words; word_idx++){
                // Pool up the probabilities of each coloring pattern
                probability_scratch.assign(num_patterns, 0.0f);
                masked_scatter_reduce(pattern_matrix[word_idx], priors,
                    probability_scratch, mask, 1.0f/static_cast<float>(words_remaining));
                // Compute the entropy via map reduce
                entropy_vector[word_idx] = map_reduce_sum(probability_scratch, &entropy);
            }
            // Find the word that maximizes entropy, obtain feedback
            candidate_idx = arg_max(entropy_vector);
        }

        feedback = word_cmp(words[candidate_idx], answer);
        // Update the number of words remaining
        for(int i = 0; i < num_words; i++){
            if(mask[i] && pattern_matrix[candidate_idx][i] != feedback){
                mask[i] = false;
                words_remaining -= 1;
            }
        }

        std::cout << "Proposed Guess: ";
        word_print(words[candidate_idx], feedback);
        std::cout << " With Entropy " << entropy_vector[candidate_idx] << "\n";
        advance_round(data, words[candidate_idx], feedback, words_remaining);
        // DEBUG CODE:
        std::cout << "Remaining Words (" << words_remaining <<"):\n";
        for(int i = 0; i < num_words; i++){
            if(mask[i]) word_print(words[i], 0, ' ');
        }
        std::cout << "\n";
        if(feedback == CORRECT_GUESS) break;
    }
}



int main(int argc, char **argv) {
    // Initialization Stage
    std::string text_filename;
    std::string prior_filename;
    int num_threads = 0;
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "f:n:p:")) != -1) {
        switch (opt) {
        case 'f':
            text_filename = optarg;
            break;
        case 'p':
            prior_filename = optarg;
            break;
        case 'n':
            num_threads = atoi(optarg);
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
    // Loading the list of words
    wordlist_t words = read_words_from_file(text_filename);
    // Loading the list of priors
    priors_t priors;
    if(empty(prior_filename))
        priors = generate_uniform_priors(words.size());
    else{
        priors = read_priors_from_file(prior_filename);
    }
    // Check if size match:
    if(priors.size() != words.size()){
        std::cerr << "Input Files Length Mismatch!\n";
        exit(1);
    }

    // Initialize all data structures used in the wordle solver
    game_data_t data;
    std::vector<std::vector<coloring_t>> 
        pattern_matrix(words.size(), std::vector<coloring_t>(words.size()));

    // Asks the user to pick a correct answer word:
    std::string buffer;
    word_t answer;
    std::cout << "Enter Final Answer:\n";
    while(1){
        std::getline(std::cin, buffer);
        if(buffer.empty()) continue;
        if (buffer == "benchmark") {
            break;
        }
        str2word(buffer, answer);
        if(!is_in_wordlist(words, answer)){
            std::cout << "The word you entered is not valid!\n";
        }
        else break;
    }
    // IO Complete
    

    // Time initialization
    auto precompute_start = std::chrono::high_resolution_clock::now();
    
    // Precompute the coloring matrix
    compute_patterns(pattern_matrix, words);

    auto precompute_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(precompute_end - precompute_start);

    std::cout << "Initialization: " << duration.count() << " milliseconds\n";

    if (buffer == "benchmark") {
        // Enter benchmark loop
        float time_total = 0.0f;
        for (auto &answer : words) {
            std::cout << "Benchmarking word: ";
            word_print(answer, 0);

            auto answer_start = std::chrono::high_resolution_clock::now();
            solver_main(data, answer, words, priors, pattern_matrix);
            auto answer_end = std::chrono::high_resolution_clock::now();
            auto answer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(answer_end - answer_start);
            time_total += answer_duration.count();
        }
        std::cout << "Time taken: " << time_total.count() << " milliseconds\n";
    } else {
        // Interactive, 1-round solver
        solver_main(data, answer, words, priors, pattern_matrix);
    }

    return 0;
}
