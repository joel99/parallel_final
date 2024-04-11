#include "wordle.h"
#include <iostream>
#include <fstream>
#include <cassert>

// Implementation of simulate_games
void simulate_games(
    bool quiet,
    bool use_empirical_value,
    const std::unordered_map<std::string, float>& priors
) {
    assert(!use_empirical_value && "Empirical value usage is not supported.");

    auto word_list = load_word_list(true); // Example call, adjust `true` based on your needs

    std::unordered_map<std::string, float> effective_priors;
    if (priors.empty()) {
        effective_priors = load_uniform_priors(word_list);
    } else {
        // Verify all words in word_list are in priors
        for (const auto& word : word_list) {
            if (priors.find(word) == priors.end()) {
                std::cerr << "Error: Word '" << word << "' not found in priors." << std::endl;
                return; // Or handle the error as appropriate
            }
        }
        effective_priors = priors;
    }

    // Check length priors loaded effectively
    assert(effective_priors.size() == word_list.size() && "Effective priors size mismatch.");
    // print general heartbeat
    if (!quiet) {
        std::cout << "Loaded " << word_list.size() << " words." << std::endl;
    }
}

// Stub for load_word_list function
std::vector<std::string> load_word_list(bool short_list) {
    // Define the file paths
    std::string data_dir = "./data/";
    std::string short_word_list_file = data_dir + "possible_words.txt";
    std::string long_word_list_file = data_dir + "allowed_words.txt";

    // Choose the file based on the short_list flag
    std::string file_path = short_list ? short_word_list_file : long_word_list_file;

    std::vector<std::string> words;
    std::string line;
    std::ifstream file(file_path);

    if (file.is_open()) {
        while (getline(file, line)) {
            // Remove newline characters and add to the list if line is not empty
            if (!line.empty()) {
                words.push_back(line);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
    }

    return words;
}

// Stub for load_uniform_priors function
std::unordered_map<std::string, float> load_uniform_priors(const std::vector<std::string>& words) {
    std::unordered_map<std::string, float> uniform_priors;
    for (const auto& word : words) {
        uniform_priors[word] = 1.0f / words.size(); // Uniform distribution
    }
    return uniform_priors;
}

int main() {
    std::cout << "Wordle Simulator and Solver" << std::endl;
    simulate_games(
        false, // quiet
        false, // use_empirical_value
        {} // priors
    );
    return 0;
}
