#include "wordle.h"
#include <iostream>
#include <cassert>

const std::string DEFAULT_FIRST_GUESS = "salet";

// Implementation of simulate_games
void simulate_games(
    bool quiet,
    bool use_empirical_value,
    const std::unordered_map<std::string, float>& priors,
    const std::string& first_guess
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

    std::string guess = first_guess;
    if (first_guess.empty()) {
        guess = get_optimal_first_guess(word_list, effective_priors);
    }

    // Ready the evaluator
    std::vector<std::string> test_set = word_list;
    std::vector<int> scores(test_set.size(), 0);
    
    size_t total_words = test_set.size();
    size_t progress = 0;
    
    for (const auto& answer : test_set) {
        print_progress_bar(++progress, total_words);

        // ... rest of your loop logic ...

        // At the end of each loop
        std::cout << std::endl; // To move to the next line after the progress bar
    }

}

int main() {
    std::cout << "Wordle Simulator and Solver" << std::endl;
    simulate_games(
        false, // quiet
        false, // use_empirical_value
        {}, // priors
        DEFAULT_FIRST_GUESS // first_guess
    );
    return 0;
}