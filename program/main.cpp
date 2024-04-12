#include "word.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstring>
#include <unistd.h>

unsigned long ceil_xdivy(unsigned long X, unsigned long Y){
    return (X + (Y - 1)) / Y;
}


void usage(char *exec_name){
    std::cout << "Usage:\n" << exec_name << "-f <word list> -n <thread count> [-p <prior probabilities>] \n";
    return;
}

int main(int argc, char **argv) {
    // Initialization Stage
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
    wordlist_t words = read_words_from_file(text_filename);
    // Loading the list of priors
    priors_t priors;
    if(empty(prior_filename))
        priors = generate_uniform_priors(words.size());
    else{
        priors = read_priors_from_file(prior_filename);
    }
    // Check if size match:
    if(priors.size() != words.size()){
        std::cerr << "Input Files Length Mismatch!\n";
        exit(1);
    }

    // Initialize game data
    game_data_t data;
    // TODO: Implement Masking



    // Simulated Wordle Game Below:
    std::string buffer;
    word_t answer;
    std::cout << "Enter Final Answer:\n";
    while(1){
        std::getline(std::cin, buffer);
        if(buffer.empty()) continue;
        str2word(buffer, answer);
        if(!is_in_wordlist(words, answer)){
            std::cout << "The word you entered is not valid!\n";
        }
        else break;
    }
    // Game Loop
    word_t guess;
    coloring_t pattern;
    unsigned long rounds = 1;
    while(1){
        std::cout << rounds << ": ";
        std::getline(std::cin, buffer);
        if(buffer.empty()) continue;
        str2word(buffer, guess);
        if(!is_in_wordlist(words, guess)){
            std::cout << "The word you entered is not valid!\n";
            continue;
        }
        // Word is valid. Check coloring
        pattern = word_cmp(guess, answer);
        advance_round(data, guess, pattern, 0);
        
        // Records Feedback
        std::cout << rounds << ": ";
        word_print(guess, pattern);

        if(pattern == CORRECT_GUESS) break;
        rounds += 1;
    }
    // Report Results:
    std::cout << "Guessed correct answer within " << rounds << " rounds! \n";


    return 0;
}
