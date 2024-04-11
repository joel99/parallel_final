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

        // * currently the shared state of the board is tracked in guesses/patterns
        std::vector<std::string> guesses(MAX_GUESSES);
        std::vector<coloring_t> patterns;
        std::vector<int> possibility_counts;
        std::vector<std::string> possibilities; // Filtered based on priors > 0 and not seen, if required.

        // Initialize possibilities...
        for (const auto& word : word_list) {
            if (effective_priors[word] > 0) {
                possibilities.push_back(word);
            }
        }

        int score = 1;
        std::string guess = first_guess;
        while (guess != answer && score <= MAX_GUESSES) {

            // print size of possibilities
            if (!quiet) {
                std::cout << std::endl; // Move to a new line after the progress bar.
                std::cout << "Size of possibilities: " << possibilities.size() << std::endl;
            }
            coloring_t pattern = get_pattern(guess, answer);
            guesses[score - 1] = guess;
            patterns.push_back(pattern);

            // JY: This is the REDUCTION step, and where we'd need to balance...
            possibilities = get_possible_words(guess, pattern, possibilities);
            possibility_counts.push_back(possibilities.size());
            score++;

            if (score <= MAX_GUESSES) guess = get_next_guess(guesses, patterns, possibilities, word_list, effective_priors);
        }

        // TODO differentiate scoring properly based on solve or not

        // Accumulate stats and build results...
        scores.push_back(score);

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
