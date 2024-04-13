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
    std::cout << "Usage:\n" << exec_name << "-f <word list> -n <thread count> [-p <prior weights>] \n";
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
                 float prior_sum,
                 std::vector<std::vector<coloring_t>> &pattern_matrix){
    // Remark: prior_sum should be returned by the prior loading function
    // Remark: This function destructively modifies the priors vector

    // Initialize Additional Solver Data
    int words_remaining = words.size();
        // The number of words consistent with the observed pattern
    int num_words = words.size();
        // The number of total candidate words
    int num_patterns = get_num_patterns();
    std::vector<float> probability_scratch;
    std::vector<float> entropy_vector(num_words, 0.0f);

    int candidate_idx;
    coloring_t feedback;

    // Computes the initial uncertainty measure
    float uncertainty = entropy_reduce(priors, prior_sum);

    // DEBUG CODE
    // std::cout << "Initial Uncertainty: " << uncertainty << "\n";

    for(int k = 0; k < 10; k ++){
        if(words_remaining <= 2){ 
            // Perform random guess if there are only 2 possible guesses left
            candidate_idx = arg_max(priors);
            entropy_vector[candidate_idx] = 0.0f;
        }
        else{
            for(int word_idx = 0; word_idx < num_words; word_idx++){
                // Pool up the weighted freqeuency of all patterns
                probability_scratch.assign(num_patterns, 0.0f);
                scatter_reduce(pattern_matrix[word_idx], priors,
                    probability_scratch);
                // Normalize the scratch work into a probability and 
                // Compute the entropy via map reduce
                entropy_vector[word_idx] = entropy_reduce(probability_scratch, 
                    prior_sum);
            }
            // Find the word that maximizes the expected entropy entropy.
            candidate_idx = arg_max(entropy_vector);
        }

        // Obtain feedback from guess
        feedback = word_cmp(words[candidate_idx], answer);
        // Perform updates to the solver data
        // Update the number of words remaining
        words_remaining = 0;
        prior_sum = 0.0f;
        for(int i = 0; i < num_words; i++){
            if(is_zero(priors[i])) continue; // prior == 0 for invalid
            if(pattern_matrix[candidate_idx][i] != feedback) priors[i] = 0.0f;
            else{
                words_remaining += 1;
                prior_sum += priors[i];
            }
        }
        // Compute the new uncertainty measure after a guess
        uncertainty = entropy_reduce(priors, prior_sum);

        // DEBUG CODE:
        std::cout << "\n Proposed Guess: ";
        word_print(words[candidate_idx], feedback);
        std::cout << "Expected Entropy from Word: " << entropy_vector[candidate_idx] << "\n";
        std::cout << "Remaining Uncertainty: " << uncertainty << "\n";
        std::cout << "Remaining Words (" << words_remaining <<"):\n";
        for(int i = 0; i < num_words; i++){
            if(!is_zero(priors[i])) word_print(words[i], 0, ' ');
        }
        std::cout << "\n";
        advance_round(data, words[candidate_idx], feedback, words_remaining);
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
    float priors_sum; // The sum of all prior weights.
    if(empty(prior_filename))
        priors = generate_uniform_priors(words.size(), priors_sum);
    else{
        priors = read_priors_from_file(prior_filename, priors_sum);
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
        str2word(buffer, answer);
        if(!is_in_wordlist(words, answer)){
            std::cout << "The word you entered is not valid!\n";
        }
        else break;
    }

    // Initialization Complete
    
    // Precompute the coloring matrix
    compute_patterns(pattern_matrix, words);

    // Enter the main solver loop
    solver_main(data, answer, words, priors, priors_sum, pattern_matrix);
    
    return 0;
}
