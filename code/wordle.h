#ifndef WORDLE_H
#define WORDLE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

// Constants
const int MIN_LETTERS = 5;
const int MAX_LETTERS = 5;
const int ALPHABET_CHARACTERS = 26;
const int COLORS = 3;

#include <unordered_map>

// Assuming WordleGameResult is a struct you'll define based on your needs
struct WordleGameResult {
    // Members to hold results of the simulation
};

// Using std::unordered_map for the 'priors' to mimic a Python dict
// The key is a std::string (short strings), and the value is a float
void simulate_games(
    bool quiet = false,
    bool use_empirical_value = false,
    const std::unordered_map<std::string, float>& priors = std::unordered_map<std::string, float>()
);

std::vector<std::string> load_word_list(bool short_list);

std::unordered_map<std::string, float> load_uniform_priors(const std::vector<std::string>& words);

#endif // WORDLE_H
