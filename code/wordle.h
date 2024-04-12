#ifndef WORDLE_H
#define WORDLE_H

#include <numeric>
#include <iostream>
#include <string>
#include <vector>
#include <cmath> // For log and isnan functions

#define WORDLEN 8 // Maximum word length
// Printing Color Codes: https://stackoverflow.com/questions/9158150/colored-output-in-c/9158263
#define RESET   "\033[0m"
#define BLACK   "\033[30m" 
#define GREEN   "\033[32m"      
#define YELLOW  "\033[33m"
#define NUMCOLORS 3 // Number of Board Colors
#define PTN_DEFAULT 0

// TODO reconcile consts
extern const bool SMOKETEST;

typedef struct word{
    char text[8];
} word_t;

// Define the equality operator for word_t
bool operator==(const word_t& lhs, const word_t& rhs) {
    return std::strncmp(lhs.text, rhs.text, sizeof(lhs.text)) == 0;
}

// Define the inequality operator for word_t (optional but good practice)
bool operator!=(const word_t& lhs, const word_t& rhs) {
    return !(lhs == rhs);
}

// Program args / flags (may be tweaked for profiling)
const bool POSSIBILITY_MASK = false; // true not implemented fully

// Constants
const int MIN_LETTERS = 5;
const int MAX_LETTERS = 5;
const int ALPHABET_CHARACTERS = 26;
const int COLORS = 3;

const int MAX_GUESSES = 6;

typedef int coloring_t; // code for possible total colorings, >= 3^MAX_LETTERS


struct GameResult { // ! not currently used
    int score;
    word_t answer;
    std::vector<word_t> guesses;
    std::vector<int> patterns;
    std::vector<int> possibilityCounts;
};

/**
 * Convert a cpp string type to the word type
 * @param source The input cpp string
 * @param dest the word_t data type to be written to
 * This function will always truncate the source string to WORDLEN size.
*/
void str2word(std::string &source, word_t &dest);

/**
 * Compare if two words are equal.
*/
bool word_eq(word_t &A, word_t &B);
bool word_eq(word_t &A, const word_t &B);
bool word_eq(const word_t &A, word_t &B);
bool word_eq(const word_t &A, const word_t &B);

/**
 * Deep Copy for the word data type
*/
void word_deepcopy(word_t &source, word_t &dest);

void word_print(word_t &word, coloring_t coloring = PTN_DEFAULT,
    char delim = '\n');

// Solving functions
coloring_t get_pattern(const word_t guess, const word_t answer);
word_t get_next_guess(
    const std::vector<word_t>& guesses, 
    const std::vector<coloring_t>& patterns,
    const std::vector<word_t>& possibilities, 
    const std::vector<word_t>& choices, 
    const std::vector<std::float_t> &priors,
    const std::vector<std::vector<coloring_t>> &coloring_matrix
);

std::vector<bool> get_possible_words_matrix(const int guess, coloring_t pattern, const std::vector<std::vector<coloring_t>>& coloring_matrix);

// More solver pieces
word_t optimal_guess(
    const std::vector<word_t> &choices,
    const std::vector<word_t> &possibilities,
    const std::vector<std::float_t> &priors,
    const std::vector<std::vector<coloring_t>> &coloring_matrix
);

std::float_t calculate_entropy(const std::vector<std::float_t>& probabilities);


// Using std::unordered_map for the 'priors' to mimic a Python dict
// The key is a word_t (short strings), and the value is a float
void simulate_games(
    std::string first_guess,
    bool quiet = false,
    bool use_empirical_value = false,
    const std::vector<float>& priors = std::vector<float>()
);

std::vector<word_t> load_word_list(bool short_list);

std::vector<float> load_uniform_priors(const std::vector<word_t>& words);

word_t get_optimal_first_guess(const std::vector<word_t>& word_list, const std::vector<float>& priors);

// Misc bells and whistle
void print_progress_bar(size_t progress, size_t total);


#endif // WORDLE_H
