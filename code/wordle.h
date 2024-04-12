#ifndef WORDLE_H
#define WORDLE_H

#include <numeric>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cmath> // For log and isnan functions

// Program args / flags (may be tweaked for profiling)
extern const std::string DEFAULT_FIRST_GUESS;
const bool POSSIBILITY_MASK = false; // true not implemented fully

// Constants
const int MIN_LETTERS = 5;
const int MAX_LETTERS = 5;
const int ALPHABET_CHARACTERS = 26;
const int COLORS = 3;

const int MAX_GUESSES = 6;

typedef struct word{
    char text[8];
} word_t;

typedef int coloring_t; // code for possible total colorings, >= 3^MAX_LETTERS


struct GameResult { // ! not currently used
    int score;
    std::string answer;
    std::vector<std::string> guesses;
    std::vector<int> patterns;
    std::vector<int> possibilityCounts;
};

// Solving functions
coloring_t get_pattern(const std::string& guess, const std::string& answer);
std::string get_next_guess(
    const std::vector<std::string>& guesses, 
    const std::vector<coloring_t>& patterns,
    const std::vector<std::string>& possibilities, 
    const std::vector<std::string>& choices, 
    const std::vector<std::float_t> &priors,
    const std::vector<std::vector<coloring_t>> &coloring_matrix
);

std::vector<std::string> get_possible_words(const std::string& guess, coloring_t pattern, const std::vector<std::string>& possibilities);
std::vector<bool> get_possible_words_matrix(const int guess, coloring_t pattern, const std::vector<std::vector<coloring_t>>& coloring_matrix);

// More solver pieces
std::string optimal_guess(
    const std::vector<std::string> &choices,
    const std::vector<std::string> &possibilities,
    const std::vector<std::float_t> &priors,
    const std::vector<std::vector<coloring_t>> &coloring_matrix
);

std::float_t calculate_entropy(const std::vector<std::float_t>& probabilities);


// Using std::unordered_map for the 'priors' to mimic a Python dict
// The key is a std::string (short strings), and the value is a float
void simulate_games(
    bool quiet = false,
    bool use_empirical_value = false,
    const std::unordered_map<std::string, float>& priors = std::unordered_map<std::string, float>(),
    const std::string& first_guess = DEFAULT_FIRST_GUESS
);

std::vector<std::string> load_word_list(bool short_list);

std::vector<float> load_uniform_priors(const std::vector<std::string>& words);

std::string get_optimal_first_guess(const std::vector<std::string>& word_list, const std::vector<float>& priors);

// Misc bells and whistle
void print_progress_bar(size_t progress, size_t total);


#endif // WORDLE_H
