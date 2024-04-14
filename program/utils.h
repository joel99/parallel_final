/**
 * Header File for I/O Functions and wordle solver data structures
 * #include "utils.h"
*/

#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <cstring>
#include "word.h"

// Word List
typedef std::vector<word_t> wordlist_t;

// The prior probabilities associated with the list of words
typedef std::vector<float> priors_t;


/**
 * Determine the wordlist index of a given word
 * @param list
 * @param word
 * @returns The index of the input word, or -1 if the word is not in the list.
*/
int list_query(wordlist_t &list, word_t& word);

/** 
 * I/O function for word list. See "utils.h" file for file formatting requirements.
*/
wordlist_t read_words_from_file(std::string input_filename);

/**
 * I/O function for test set. Must read the list of possible words first.
 * (The test set is the vector of indices of answer words)
*/
std::vector<int> read_test_set_from_file(std::string input_filename, wordlist_t possible_words);

/**
 * I/O function for prior list. See "utils.h" file for file formatting requirements.
 * @param sum returns the sum of all priors
*/
priors_t read_priors_from_file(std::string input_filename, float &sum);

/**
 * Generate a uniform prior if prior file is not provided.
 * @param sum returns the sum of all priors
*/
priors_t generate_uniform_priors(unsigned long wordlist_size, float &sum);

/**
 * Generate random prior weights
 * @param sum returns the sum of all generated priors
 * @param lo lower bound of weights
 * @param hi upper bound of weights
*/
priors_t generate_random_priors(unsigned long size, float &sum, 
    float lo = 1e-6f, float hi = 1.0f);

/**
 * File Formatting:
 * <Header> - The number of words in both the prior file and the word file
 * <Content>
*/

/**
 * Wordle Game Record Data
*/

struct data_entry{
    int guess;               // The index of the guessed word
    unsigned int remaining;  // The number of words remaining
    float uncertainty;       // 
    coloring_t pattern;      // The board pattern for this guess
};

// Initialize game data by simply defining game_data_t
typedef std::vector<struct data_entry> game_data_t;

/**
 * Advance to the next round in wordle game simulation
 * @param data Game data record
 * @param guess The index to the input guess word
 * @param the pattern of this round
 * @param words_remaining The number of words remaining after this round
*/
void advance_round(game_data_t &data, int &guess, coloring_t pattern,
    unsigned int words_remaining, float remaining_uncertainty);
 

/**
 * Used in verbose mode: Reports the final game statistics
*/
void report_game(game_data_t &data, wordlist_t &words);



#endif /* UTILS_H */