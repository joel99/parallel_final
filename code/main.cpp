#include "wordle.h"
#include <iostream>
#include <cassert>

const bool SMOKETEST = true; // quick debuggin
// const bool SMOKETEST = false; // quick debuggin

// Implementation of simulate_games
void simulate_games(
    std::string first_guess,
    bool quiet,
    bool use_empirical_value,
    const std::vector<float>& priors
) {
    assert(!use_empirical_value && "Empirical value usage is not supported.");

    auto word_list = load_word_list(SMOKETEST); // FAST MODE

    std::vector<float> effective_priors;
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

    word_t init_guess;
    if (first_guess.empty()) {
        init_guess = get_optimal_first_guess(word_list, effective_priors);
    } else {
        str2word(first_guess, init_guess);
    }
    exit(0);
    // Ready the evaluator
    std::vector<word_t> test_set = word_list;
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
        if (answer.text != "annex") { // DEBUG: Skip to speed up
            continue;
        }
        print_progress_bar(++progress, total_words);
        // print true word
        if (!quiet) {
            std::cout << "Answer: " << answer.text << std::endl;
            // check whether the answer is in the word list
            if (std::find(word_list.begin(), word_list.end(), answer) == word_list.end()) {
                std::cout << "Answer not in word list." << std::endl;
            }
        }

        // * currently the shared state of the board is tracked in guesses/patterns
        std::vector<word_t> guesses(MAX_GUESSES);
        std::vector<coloring_t> patterns;
        std::vector<int> possibility_counts;
        std::vector<word_t> possibilities; // Filtered based on priors > 0 and not seen, if required.
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
        word_t guess = init_guess;
        
        while (!word_eq(guess, answer) && score <= MAX_GUESSES) {
            
            if (answer.text == "annex") {
            // if (guess == "aback" && answer == "annex") {
                std::cout << "Pattern between guess and 'annex': " << get_pattern(guess, answer) << std::endl;
                int guess_idx = std::distance(word_list.begin(), std::find(word_list.begin(), word_list.end(), guess));
                int answer_og_idx = std::distance(word_list.begin(), std::find(word_list.begin(), word_list.end(), answer));
                int answer_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), answer));
                std::cout << "Coloring matrix og: " << coloring_matrix[guess_idx][answer_og_idx] << std::endl;
                std::cout << "Coloring matrix in working index: " << working_coloring_matrix[guess_idx][answer_idx] << std::endl;
                // Things have gone wrong if there's a mismatch
                word_print(word_list[guess_idx], get_pattern(word_list[guess_idx], word_list[answer_og_idx]));
            }

            coloring_t pattern = get_pattern(guess, answer);
            guesses[score - 1] = guess;
            patterns.push_back(pattern);
            // guess idx should index source word list / legal word list, not possibilities for answers
            // prevention of repeat guesses should be implemented here
            int guess_idx = std::distance(word_list.begin(), std::find(word_list.begin(), word_list.end(), guess));

            // print size of possibilities
            if (!quiet) {
                std::cout << std::endl; // Move to a new line after the progress bar.
                std::cout << "Turn: " << score << " Guess: " << guess.text << " at index: " << guess_idx << std::endl;
                std::cout << "Size of possibilities: " << possibilities.size() << std::endl;
                // OK, salet is not in here... where ... why?
                // Check size of possibilites and identified coloring_matrix
                std::cout << "Size of row 0 of coloring_matrix: " << working_coloring_matrix[0].size() << std::endl;
                std::cout << "Size of coloring_matrix: " << working_coloring_matrix.size() << std::endl;
            }

            // TODO use valid mask here
            std::vector<bool> mask = get_possible_words_matrix(guess_idx, pattern, working_coloring_matrix);
            // print mask, guess, pattern
            // if (!quiet) {
            //     std::cout << "Mask: ";
            //     for (int i = 0; i < mask.size(); i++) {
            //         std::cout << mask[i] << " ";
            //     }
            //     std::cout << std::endl;
            // }

            // print sum anyway
            // if (!quiet) {
            //     int mask_sum = std::accumulate(mask.begin(), mask.end(), 0);
            //     std::cout << "Sum of mask: " << mask_sum << std::endl;
            // }
            if (POSSIBILITY_MASK) {
                // just keep the mask and call mask-based functions
                possibility_counts.push_back(std::accumulate(mask.begin(), mask.end(), 0));
            } else {
                // First check what index the answer is initially
                // print answer index
                // if (!quiet) {
                //     int answer_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), answer));
                //     std::cout << "Answer index: " << answer_idx << std::endl;
                // }
                std::vector<word_t> new_possibilities;
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
                // if (!quiet) {
                    // int new_answer_idx = std::distance(possibilities.begin(), std::find(possibilities.begin(), possibilities.end(), answer));
                    // std::cout << "New answer index: " << new_answer_idx << std::endl;
                // }

                // print possibilities if length < 10
                // if (!quiet && possibilities.size() < 200) {
                //     std::cout << "Possibilities: ";
                //     for (int i = 0; i < possibilities.size(); i++) {
                //         std::cout << possibilities[i] << " ";
                //     }
                //     std::cout << std::endl;
                // }
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
        if (!quiet) {
            std::cout << "Score: " << score << std::endl;
        }

        // TODO differentiate scoring properly based on solve or not

        // Accumulate stats and build results...
        scores.push_back(score);

        // At the end of each loop
        std::cout << std::endl; // To move to the next line after the progress bar

        // EXIT THE PRGOGRAM (SMOKETEST)
    }

}

int main() {
    std::cout << "Wordle Simulator and Solver" << std::endl;
    std::string first_guess = "steam";
    simulate_games(
        first_guess,
        false, // quiet
        false // use_empirical_value
    );
    return 0;
}
