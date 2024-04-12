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
#include "word.h"

// Word List
typedef std::vector<word_t> wordlist_t;

// The prior probabilities associated with the list of words
typedef std::vector<float> priors_t;

/** 
 * I/O function for word list. See "utils.h" file for file formatting requirements.
*/
wordlist_t read_words_from_file(std::string input_filename);

/**
 * I/O function for prior list. See "utils.h" file for file formatting requirements.
*/

priors_t read_priors_from_file(std::string input_filename);

/**
 * Generate a uniform prior if prior file is not provided.
*/
priors_t generate_uniform_priors(unsigned long wordlist_size);

/**
 * File Formatting:
 * <Header> - The number of words in both the prior file and the word file
 * <Content>
*/

/**
 * Wordle Game Record Data
*/

struct data_entry{
    word_t guess;            // The final guessed word in this round
    coloring_t pattern;      // The board pattern for this guess
    unsigned int remaining; // How many valid words are remaining after this guess
};

// Initialize game data by simply defining game_data_t
typedef std::vector<struct data_entry> game_data_t;

/**
 * Advance to the next round in wordle game simulation
 * @param data Game data record
 * @param guess The input guess word
 * @param the pattern of this round
 * @param words_remaining The number of words remaining after this round
*/
void advance_round(game_data_t &data, word_t &guess, coloring_t pattern,
    unsigned int words_remaining);
 
unsigned long report_game_iterations(game_data_t &data);


// Debugging Functions
bool is_in_wordlist(wordlist_t &list, word_t& word);


#endif /* UTILS_H */