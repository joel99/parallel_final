// Contains the solver logic.

#include "wordle.h"

coloring_t get_pattern(const std::string& guess, const std::string& answer) {
    /* 
        Returns coloring_t.
    */
    // return 0; // Assume 0 is a placeholder pattern
    // Indicator array, prevent matching the same letter twice. Init'd to false
    bool query_matched[MAX_LETTERS] = { 0 };
    bool answer_matched[MAX_LETTERS] = { 0 };

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for(int i = 0; i < MAX_LETTERS; i++){
        if(guess[i] == answer[i]){ // todo restore word_t struct
            out += (2 * mult);
            query_matched[i] = true;
            answer_matched[i] = true;
        }
        mult *= 3;
    }

    // reset multiplier
    mult = 1;
    // Check for yellow boxes
    for(int i = 0; i < MAX_LETTERS; i++){ // query index
        if(query_matched[i]) {
            mult *= 3;
            continue;
        }
        for(int j = 0; j < MAX_LETTERS; j++){// answer index
            if(i == j || answer_matched[j]) continue;
            if(guess[i] == answer[j]){
                out += mult;
                query_matched[i] = true;
                answer_matched[j] = true;
                break;
            }
        }
        mult *= 3;
    }
    return out;
}

std::string get_next_guess(const std::vector<std::string>& guesses, const std::vector<int>& patterns, const std::vector<std::string>& possibilities) {
    // Placeholder implementation
    if (!possibilities.empty()) {
        return possibilities.front(); // Simply return the first possibility as a placeholder
    }
    return ""; // Return an empty string if no possibilities are left
}

std::vector<std::string> get_possible_words(const std::string& guess, coloring_t pattern, const std::vector<std::string>& possibilities) {
    // TODO replace with coloring matrix subset op, coloring.where(coloring[guess] == pattern)
    // Basic serial implementation
    std::vector<std::string> filteredWords;
    for (const auto& word : possibilities) {
        if (get_pattern(guess, word) == pattern) {
            filteredWords.push_back(word);
        }
    }
    return filteredWords;
}