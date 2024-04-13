#include "utils.h"

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

/*******************************
 * Game data Functions
********************************/


void advance_round(game_data_t &data, word_t &guess, coloring_t pattern,
    unsigned int words_remaining){
    struct data_entry new_entry;
    word_deepcopy(guess, new_entry.guess);
    new_entry.remaining = words_remaining;
    new_entry.pattern = pattern;
    data.push_back(new_entry);
}

unsigned long report_game_iterations(game_data_t &data){
    return data.size();
}


// Debugging Functions
bool is_in_wordlist(wordlist_t &list, word_t& word){
    for (word_t &w : list){
        if(word_eq(w, word)) return true;
    }
    return false;
}