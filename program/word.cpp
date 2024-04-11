#include "word.h"

coloring_t word_cmp(word_t &query, word_t &answer){
    // Indicator array, prevent matching the same letter twice. Init'd to false
    bool query_matched[WORDLEN] = { 0 };
    bool answer_matched[WORDLEN] = { 0 };

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for(int i = 0; i < WORDLEN; i++){
        if(query.text[i] == answer.text[i]){
            out += (2 * mult);
            query_matched[i] = true;
            answer_matched[i] = true;
        }
        mult *= 3;
    }

    // reset multiplier
    mult = 1;
    // Check for yellow boxes
    for(int i = 0; i < WORDLEN; i++){ // query index
        if(query_matched[i]) {
            mult *= 3;
            continue;
        }
        for(int j = 0; j < WORDLEN; j++){// answer index
            if(i == j || answer_matched[j]) continue;
            if(query.text[i] == answer.text[j]){
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

void color_check(coloring_t coloring, word_t &query, word_t &answer){
    for(int i = 0; i < WORDLEN; i ++)
        std::cout << query.text[i];
    std::cout << "\n";

    for(int i = 0; i < WORDLEN; i ++){
        if(coloring % 3 == 2)
            std::cout << "g";
        else if (coloring % 3 == 1)
            std::cout << "y";
        else
            std::cout << "-";
        coloring = coloring / 3;
    }

    std::cout << "\n";
    for(int i = 0; i < WORDLEN; i ++)
        std::cout << answer.text[i];
    std::cout << "\n";
}