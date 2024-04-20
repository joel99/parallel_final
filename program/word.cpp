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
    // Optimized with some 15-213 data lab magic
    int query_matched = 0x0;
    int mask = 0x1;

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for(int i = 0; i < wordlen; i++){
        // printf("%x, %x\n", mask, query_matched);
        if(query.text[i] == answer.text[i]){
            out += (2 * mult);
            query_matched |= mask;
        }
        mask <<= 1;
        mult *= NUMCOLORS;
    }

    // reset multiplier and mask
    mask = 0x1;
    mult = 1;
    // Add a new lane mask indicating if answer is matched
    int answer_matched = query_matched;
    // Check for yellow boxes
    for(int i = 0; i < wordlen; i++){ // query index
        if(query_matched & mask) {
            mult *= NUMCOLORS;
            mask <<= 1;
            continue;
        }
        int answer_mask = 0x1;
        for(int j = 0; j < wordlen; j++){// answer index
            // printf("%x, %x, %x\n", answer_mask, answer_matched, answer_matched & answer_mask);
            if(!(answer_matched & answer_mask) && query.text[i] == answer.text[j]){
                out += mult;
                query_matched |= mask;
                answer_matched |= answer_mask;
                break;
            }
            answer_mask <<= 1;
        }
        mult *= NUMCOLORS;
        mask <<= 1;
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