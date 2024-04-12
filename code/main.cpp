#include "wordle.h"
#include <iostream>
#include <cassert>

const std::string DEFAULT_FIRST_GUESS = "steam"; // more common, salet is not in common list and is slower to iterate atm
const bool SMOKETEST = true; // quick debuggin
// const bool SMOKETEST = false; // quick debuggin
// const std::string DEFAULT_FIRST_GUESS = "salet";

// Implementation of simulate_games
void simulate_games(
    bool quiet,
    bool use_empirical_value,
    const std::unordered_map<std::string, float>& priors,
    const std::string& first_guess
) {
    assert(!use_empirical_value && "Empirical value usage is not supported.");

    auto word_list = load_word_list(SMOKETEST); // FAST MODE

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

    std::vector<std::vector<coloring_t>> coloring_matrix; // Shape Guess x Possibility
    for (int i = 0; i < word_list.size(); i++) {
        std::vector<coloring_t> row = std::vector<coloring_t>(word_list.size());
        for (int j = 0; j < word_list.size(); j++) {
            row[j] = get_pattern(word_list[i], word_list[j]);
        }
        coloring_matrix.push_back(row);
    }
    
    for (const auto& answer : test_set) {
        print_progress_bar(++progress, total_words);
        // print true word
        if (!quiet) {
            std::cout << "Answer: " << answer << std::endl;
            // check whether the answer is in the word list
            if (std::find(word_list.begin(), word_list.end(), answer) == word_list.end()) {
                std::cout << "Answer not in word list." << std::endl;
            }
        }

        // * currently the shared state of the board is tracked in guesses/patterns
        std::vector<std::string> guesses(MAX_GUESSES);
        std::vector<coloring_t> patterns;
        std::vector<int> possibility_counts;
        std::vector<std::string> possibilities; // Filtered based on priors > 0 and not seen, if required.
        std::vector<float> working_priors;
        std::vector<std::vector<coloring_t>> working_coloring_matrix = coloring_matrix; // Shape Guess x Possibility
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
            int guess_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), guess));
            // print guess and index
            if (!quiet) {
                // OK, salet is not in here... where ... why?
                std::cout << "Turn: " << score << " Guess: " << guess << " at index: " << guess_idx << std::endl;
                // Check size of possibilites and identified coloring_matrix
                std::cout << "Size of possibilities: " << possibilities.size() << std::endl;
                std::cout << "Size of row 0 of coloring_matrix: " << working_coloring_matrix[0].size() << std::endl;
                std::cout << "Size of coloring_matrix: " << working_coloring_matrix.size() << std::endl;
            }

            // TODO use valid mask here
            std::vector<bool> mask = get_possible_words_matrix(guess_idx, pattern, working_coloring_matrix);
            // print sum anyway
            if (!quiet) {
                int mask_sum = std::accumulate(mask.begin(), mask.end(), 0);
                std::cout << "Sum of mask: " << mask_sum << std::endl;
            }
            if (POSSIBILITY_MASK) {
                // just keep the mask and call mask-based functions
                possibility_counts.push_back(std::accumulate(mask.begin(), mask.end(), 0));
            } else {
                // reduction

                // First check what index the answer is initially
                int answer_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), answer));
                // print answer index
                if (!quiet) {
                    std::cout << "Answer index: " << answer_idx << std::endl;
                }
                std::vector<std::string> new_possibilities;
                std::vector<float> new_priors;
                std::vector<std::vector<coloring_t>> new_coloring_matrix;
                // Init
                for (int guess = 0; guess < coloring_matrix.size(); guess++) {
                    new_coloring_matrix.push_back(std::vector<coloring_t>());
                }
                for (int i = 0; i < mask.size(); i++) {
                    if (mask[i]) {
                        new_possibilities.push_back(possibilities[i]);
                        new_priors.push_back(working_priors[i]);
                        for (int j = 0; j < coloring_matrix.size(); j++) {
                            new_coloring_matrix[j].push_back(coloring_matrix[j][i]);
                        }
                    }
                }
                possibilities = new_possibilities;
                working_priors = new_priors;
                working_coloring_matrix = new_coloring_matrix;
                possibility_counts.push_back(possibilities.size());

                // Now check new index
                int new_answer_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), answer));
                if (!quiet) {
                    std::cout << "New answer index: " << new_answer_idx << std::endl;
                }

                // print possibilities if length < 10
                if (!quiet && possibilities.size() < 10) {
                    std::cout << "Possibilities: ";
                    for (int i = 0; i < possibilities.size(); i++) {
                        std::cout << possibilities[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }

            score++;

            if (score <= MAX_GUESSES) guess = get_next_guess(
                guesses, 
                patterns, 
                possibilities, 
                word_list, 
                working_priors,
                working_coloring_matrix
            );
        }

        // TODO differentiate scoring properly based on solve or not

        // Accumulate stats and build results...
        scores.push_back(score);

        // At the end of each loop
        std::cout << std::endl; // To move to the next line after the progress bar

        // EXIT THE PRGOGRAM (SMOKETEST)
        if (SMOKETEST) {
            break;
        }
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
