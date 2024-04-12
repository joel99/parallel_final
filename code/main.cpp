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

    std::vector<float> effective_priors; // todo convert to vector of flaots
    if (priors.empty()) {
        effective_priors = load_uniform_priors(word_list);
    } else {
        // Verify all words in word_list are in priors
        assert(priors.size() == word_list.size() && "Priors size mismatch.");
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

    std::vector<std::vector<coloring_t>> coloring_matrix; // ! Not currently used
    for (int i = 0; i < word_list.size(); i++) {
        std::vector<coloring_t> row = std::vector<coloring_t>(word_list.size());
        for (int j = 0; j < word_list.size(); j++) {
            row[j] = get_pattern(word_list[i], word_list[j]);
        }
    }
    
    for (const auto& answer : test_set) {
        print_progress_bar(++progress, total_words);

        // * currently the shared state of the board is tracked in guesses/patterns
        std::vector<std::string> guesses(MAX_GUESSES);
        std::vector<coloring_t> patterns;
        std::vector<int> possibility_counts;
        std::vector<std::string> possibilities; // Filtered based on priors > 0 and not seen, if required.
        std::vector<float> working_priors;

        // V1: TO CONSIDER Instead of reducing the active set of words, we simply maintain a mask
        if (POSSIBILITY_MASK) {
            std::vector<bool> valid_mask(word_list.size(), true); // Mask to track valid words (not seen yet
            for (int i = 0; i < word_list.size(); i++) {
                if (effective_priors[i] == 0) {
                    valid_mask[i] = false;
                }
                possibilities.push_back(word_list[i]);
                working_priors.push_back(effective_priors[i]);
            }
        } else {
            for (int i = 0; i < word_list.size(); i++) {
                if (effective_priors[i] > 0) {
                    possibilities.push_back(word_list[i]);
                    working_priors.push_back(effective_priors[i]);
                }
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

            // TODO: JY -> SH: there's a dramatic reduction in number of possibilities, suspiciously high. Can you check if the reduction is happening correctly?

            // JY: This is the REDUCTION step, and where we'd need to balance...
            if (POSSIBILITY_MASK) {
                int guess_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), guess));
                std::vector<bool> mask = get_possible_words_matrix(guess_idx, pattern, coloring_matrix);
                int count = 0;
                
            } else {
                possibilities = get_possible_words(guess, pattern, possibilities); // Not supported because it doesn't reduce prior/coloring in tandem
                // TODO not implemented further than this
                possibility_counts.push_back(possibilities.size());
            }

            score++;

            if (score <= MAX_GUESSES) guess = get_next_guess(
                guesses, 
                patterns, 
                possibilities, 
                word_list, 
                working_priors,
                coloring_matrix
            );
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
