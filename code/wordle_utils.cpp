// Contains utility functions like loading word lists and priors.
#include "wordle.h"
#include <fstream>

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

std::vector<float> load_uniform_priors(const std::vector<std::string>& words) {
    std::vector<float> uniform_priors(words.size(), 1.0 / words.size());
    return uniform_priors;
}

std::string get_optimal_first_guess(const std::vector<std::string>& word_list, const std::vector<float>& priors) {
    // Placeholder logic for finding the optimal first guess
    // For now, simply return the default first guess
    return DEFAULT_FIRST_GUESS;
}

void print_progress_bar(size_t progress, size_t total) {
    const int bar_width = 70;
    float percent = static_cast<float>(progress) / total;
    std::cout << "[";
    int pos = bar_width * percent;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(percent * 100.0) << " %\r";
    std::cout.flush();
}


float calculate_entropy(const std::vector<float>& probabilities) {
    float entropy = 0.0;
    for (float p : probabilities) {
        if (p > 0) {  // Only add to the entropy if p is non-zero
            entropy += p * log(p);
        }
    }
    return -entropy;  // The entropy formula has a negative sign
}