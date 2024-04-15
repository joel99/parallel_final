#include "word.h"

// Printing Color Codes: https://stackoverflow.com/questions/9158150/colored-output-in-c/9158263
#define RESET   "\033[0m"
#define BLACK   "\033[30m" 
#define GREEN   "\033[32m"      
#define YELLOW  "\033[33m"

unsigned long get_num_patterns(){
    unsigned long out = 1;
    for(int i = 0; i < wordlen; i++)
        out *= NUMCOLORS;
    return out;
}

bool is_correct_guess(coloring_t c){
    return (c == get_num_patterns() - 1);
}

coloring_t word_cmp(word_t &query, word_t &answer){
    // Indicator array, prevent matching the same letter twice. Init'd to false
    bool query_matched[MAXLEN] = { 0 }; // Allocate enough space.
    bool answer_matched[MAXLEN] = { 0 };

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for(int i = 0; i < wordlen; i++){
        if(query.text[i] == answer.text[i]){
            out += (2 * mult);
            query_matched[i] = true;
            answer_matched[i] = true;
        }
        mult *= NUMCOLORS;
    }

    // reset multiplier
    mult = 1;
    // Check for yellow boxes
    for(int i = 0; i < wordlen; i++){ // query index
        if(query_matched[i]) {
            mult *= NUMCOLORS;
            continue;
        }
        for(int j = 0; j < wordlen; j++){// answer index
            if(i == j || answer_matched[j]) continue;
            if(query.text[i] == answer.text[j]){
                out += mult;
                query_matched[i] = true;
                answer_matched[j] = true;
                break;
            }
        }
        mult *= NUMCOLORS;
    }
    return out;
}

void str2word(std::string &source, word_t &dest){
    strncpy(dest.text, source.c_str(), wordlen);
}

void word_deepcopy(word_t &source, word_t &dest){
    strncpy(dest.text, source.text, wordlen);
}

bool word_eq(word_t &A, word_t &B){
    return(strncmp(A.text, B.text, wordlen) == 0);
}

void word_print(word_t &word, coloring_t coloring, char delim){
    for(int i = 0; i < wordlen; i ++){
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