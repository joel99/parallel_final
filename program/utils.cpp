#include "utils.h"
#include "mathutils.h"

int list_query(wordlist_t &list, word_t& word){
    int n = list.size();
    for(int i = 0; i < n; i++){
        if(word_eq(list[i], word)) return i;
    }
    return -1;
}

/*******************************
 * File I/O Functions
********************************/

wordlist_t read_words_from_file(std::string input_filename){
    std::ifstream file(input_filename);
    if(!file.is_open()){
        std::cerr << "Unable to open file: " << input_filename << " .\n";
        exit(1);
    }
    std::string line;

    unsigned long word_count;
    unsigned long word_index = 0;
    getline(file, line);
    if(line.empty()){
        std::cerr << "Unsupported File Format: " << input_filename 
            << " [Err: Word Count]\n";
        exit(1);
    }
    // Attempt to read the word_count parameter.
    try{
       word_count = std::stoul(line);
    }
    catch(const std::exception& e){
        std::cerr << "Unsupported File Format: " << input_filename
             << " [Err: Word Count]\n";
        exit(1);
    }

    wordlist_t out(word_count);
    while (getline(file, line)) {
        if(word_index >= word_count) break; // Avoid buffer overflow.
        // Remove newline characters and add to the list if line is not empty
        if (!line.empty()) {
            str2word(line, out[word_index]);
            word_index ++;
        }
    }
    file.close();
    return out;
}


priors_t read_priors_from_file(std::string input_filename, float &sum){
    std::ifstream file(input_filename);
    if(!file.is_open()){
        std::cerr << "Unable to open file: " << input_filename << " .\n";
        exit(1);
    }
    std::string line;

    unsigned long count;
    unsigned long index = 0;
    getline(file, line);
    if(line.empty()){
        std::cerr << "Unsupported File Format: " << input_filename 
            << " [Err: Prior Count]\n";
        exit(1);
    }
    // Attempt to read the word_count parameter.
    try{
       count = std::stoul(line);
    }
    catch(const std::exception& e){
        std::cerr << "Unsupported File Format: " << input_filename
             << " [Err: Prior Count]\n";
        exit(1);
    }
    priors_t out(count);
    sum = 0.0f;
    float temp;
    while (getline(file, line)) {
        if(index >= count) break; // Avoid buffer overflow.
        // Remove newline characters and add to the list if line is not empty
        if (!line.empty()) {
            try{
                temp = std::stof(line);
            }
            catch(const std::exception& e){
                std::cerr << "Unsupported File Format: " << input_filename
                    << " [Err: Invalid Prior]\n";
                 exit(1);
            }
            out[index] = temp;
            sum += temp;
            index ++;
        }
    }
    file.close();
    return out;
}

priors_t generate_uniform_priors(unsigned long size, float &sum){
    priors_t out(size, 1.0f);
    sum = 1.0f * static_cast<float>(size);
    return out;
}

priors_t generate_random_priors(unsigned long size, float &sum,
    float lo, float hi){
    priors_t out(size, 1.0f);
    float gen;
    sum = 0.0f;
    for(unsigned long i = 0; i < size; i++){
        gen = f_rand(lo, hi);
        out[i] = gen;
        sum += gen;
    }
    return out;
}

std::vector<int> read_test_set_from_file(std::string input_filename, wordlist_t possible_words){
    std::ifstream file(input_filename);
    if(!file.is_open()){
        std::cerr << "Unable to open file: " << input_filename << " .\n";
        exit(1);
    }
    std::string line;

    unsigned long word_count;
    getline(file, line);
    if(line.empty()){
        std::cerr << "Unsupported File Format: " << input_filename 
            << " [Err: Word Count]\n";
        exit(1);
    }
    // Attempt to read the word_count parameter.
    try{
       word_count = std::stoul(line);
    }
    catch(const std::exception& e){
        std::cerr << "Unsupported File Format: " << input_filename
             << " [Err: Word Count]\n";
        exit(1);
    }

    std::vector<int> out(word_count);
    unsigned long list_index = 0;
    word_t buffer;
    while (getline(file, line)) {
        if(list_index >= word_count) break; // Avoid buffer overflow.
        if (!line.empty()) {
            str2word(line, buffer);
            out[list_index] = list_query(possible_words, buffer);
            if(out[list_index] < 0){
                // Safety Check: Check if the test set contains illegal words
                std::cerr << "Test Set " << input_filename << " contains illegal word ";
                word_print(buffer);
                exit(1);
            }
            list_index ++;
        }
    }
    file.close();
    return out;
}

/*******************************
 * Game data Functions (Depricated)
********************************/


void advance_round(game_data_t &data, int &guess, coloring_t pattern,
    unsigned int words_remaining, float remaining_uncertainty){
    struct data_entry new_entry;
    new_entry.guess = guess;
    new_entry.pattern = pattern;
    new_entry.remaining = words_remaining;
    new_entry.uncertainty = remaining_uncertainty;
    data.push_back(new_entry);
}
 

/**
 * Used in verbose mode: Reports the final game statistics
*/
void report_game(game_data_t &data, wordlist_t &words){
    return;
}


