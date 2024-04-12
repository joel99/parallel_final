// C++ script for unit tests for different functions. This is not production code.

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include "word.h"

int main(int argc, char **argv){
    std::string query_buf;
    std::string answer_buf;
    word_t query;
    word_t answer;
    while(query_buf.empty()){
        std::cout << "Enter Query Word:\n";
        std::getline(std::cin, query_buf);
    }
    if(query_buf.length() > WORDLEN) std::cout << "Truncated excessive input\n";
    for(unsigned int i = 0; i < WORDLEN; i++){
        if(i < query_buf.length())
            query.text[i] = query_buf[i];
        else
            query.text[i] = 0x20;
    }

    while(answer_buf.empty()){
        std::cout << "Enter Answer Word:\n";
        std::getline(std::cin, answer_buf);
    }
    if(answer_buf.length() > WORDLEN) std::cout << "Truncated excessive input\n";
    for(unsigned int i = 0; i < WORDLEN; i++){
        if(i < answer_buf.length())
            answer.text[i] = answer_buf[i];
        else
            answer.text[i] = 0x20;
    }

    coloring_t result = word_cmp(query, answer);
    // color_check(result, query, answer);
    return 0;
}