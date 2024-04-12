// Contains utility functions like loading word lists and priors.
#include "wordle.h"
#include <fstream>

coloring_t get_pattern(const word_t &guess, const word_t &answer)
{
    /*
        Returns coloring_t.
    */
    // return 0; // Assume 0 is a placeholder pattern
    // Indicator array, prevent matching the same letter twice. Init'd to false
    bool query_matched[MAX_LETTERS] = {0};
    bool answer_matched[MAX_LETTERS] = {0};

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for (int i = 0; i < MAX_LETTERS; i++)
    {
        if (guess.text[i] == answer.text[i])
        {
            out += (2 * mult);
            query_matched[i] = true;
            answer_matched[i] = true;
        }
        mult *= 3;
    }

    // reset multiplier
    mult = 1;
    // Check for yellow boxes
    for (int i = 0; i < MAX_LETTERS; i++)
    { // query index
        if (query_matched[i])
        {
            mult *= 3;
            continue;
        }
        for (int j = 0; j < MAX_LETTERS; j++)
        { // answer index
            if (i == j || answer_matched[j])
                continue;
            if (guess.text[i] == answer.text[j])
            {
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

void str2word(std::string &source, word_t &dest){
    strncpy(dest.text, source.c_str(),WORDLEN);
}

void word_deepcopy(word_t &source, word_t &dest){
    strncpy(dest.text, source.text, WORDLEN);
}


bool word_eq(const word_t &A, const word_t &B){
    return(strncmp(A.text, B.text, WORDLEN) == 0);
}

void word_print(word_t &word, coloring_t coloring, char delim){
    for(int i = 0; i < WORDLEN; i ++){
        if(coloring % 3 == 2)
            std::cout << GREEN << word.text[i] << RESET;
        else if (coloring % 3 == 1)
            std::cout << YELLOW << word.text[i] << RESET;
        else
            std::cout << BLACK << word.text[i] << RESET;
        coloring = coloring / NUMCOLORS;
    }
    std::cout << delim;
}

std::vector<word_t> load_word_list(bool short_list) {
    // Define the file paths
    std::string data_dir = "./data/";
    std::string short_word_list_file = data_dir + "possible_words.txt";
    std::string long_word_list_file = data_dir + "allowed_words.txt";

    // Choose the file based on the short_list flag
    std::string file_path = short_list ? short_word_list_file : long_word_list_file;

    std::vector<word_t> words;
    std::string line;
    word_t coerce;
    std::ifstream file(file_path);

    if (file.is_open()) {
        while (getline(file, line)) {
            // Remove newline characters and add to the list if line is not empty
            if (!line.empty()) {
                str2word(line, coerce);
                words.push_back(coerce);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << file_path << std::endl;
    }

    return words;
}

std::vector<float> load_uniform_priors(const std::vector<word_t>& words) {
    std::vector<float> uniform_priors(words.size(), 1.0 / words.size());
    return uniform_priors;
}

word_t get_optimal_first_guess(const std::vector<word_t>& word_list, const std::vector<float>& priors) {
    // Placeholder logic for finding the optimal first guess
    // For now, simply return the default first guess
    return word_list[0];
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

std::float_t calculate_entropy(const std::vector<std::float_t>& probabilities) {
    float entropy = 0.0;
    for (float p : probabilities) {
        if (p > 0) {  // Only add to the entropy if p is non-zero
            entropy += p * log(p);
        }
    }
    return -entropy;  // The entropy formula has a negative sign
}