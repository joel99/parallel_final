#include "word.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstring>
#include <unistd.h>

// Data Strucutre Abstractions
typedef std::vector<float> priors_t;

void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << "-f <word list> -n <thread count> [-p <prior probabilities>] \n";
    return;
}

wordlist_t read_words(std::string input_filename) {

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
    wordlist_t words(word_count);
    while (getline(file, line)) {
        if(word_index >= word_count) break; // Avoid buffer overflow.
        // Remove newline characters and add to the list if line is not empty
        if (!line.empty()) {
            str2word(line, words[word_index]);
            word_index ++;
        }
    }
    file.close();
    return words;
}


priors_t generate_uniform_priors(unsigned long size){
    priors_t out(size, 1.0f / size);
    return out;
}

priors_t read_priors(std::string input_filename){
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
    while (getline(file, line)) {
        if(index >= count) break; // Avoid buffer overflow.
        // Remove newline characters and add to the list if line is not empty
        if (!line.empty()) {
            try{
                out[index] = std::stof(line);
            }
            catch(const std::exception& e){
                std::cerr << "Unsupported File Format: " << input_filename
                    << " [Err: Invalid Prior]\n";
                 exit(1);
            }
            index ++;
        }
    }
    file.close();
    return out;
}

int main(int argc, char **argv) {
    std::string text_filename;
    std::string prior_filename;
    int num_threads = 0;
    int opt;
    // Read program parameters
    while ((opt = getopt(argc, argv, "f:n:p:")) != -1) {
        switch (opt) {
        case 'f':
            text_filename = optarg;
            break;
        case 'p':
            prior_filename = optarg;
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }
    if(empty(text_filename) || num_threads <= 0){
        usage(argv[0]);
        exit(1);
    }
    // Loading the list of words
    wordlist_t words = read_words(text_filename);
    // Loading the list of priors
    priors_t priors;
    if(empty(prior_filename))
        priors = generate_uniform_priors(words.size());
    else{
        priors = read_priors(prior_filename);
    }
    // Check if size match:
    if(priors.size() != words.size()){
        std::cerr << "Prior file length differs from word file length.\n";
        exit(1);
    }

    for (word_t &word : words){
        word_print(word);
    }
    for (auto &i:priors){
        std::cout << i << "\n";
    }

    
    return 0;
}
