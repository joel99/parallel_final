/**
 * Header File for Word Data Structure and Word wise Operations
 * #include "word.h"
*/

#ifndef WORD_H
#define WORD_H

#include <iostream>
#include <cstring>

#define WORDLEN 8 // Maximum word length
#define NUMCOLORS 3 // Number of Board Colors
#define PTN_DEFAULT 0
#define CORRECT_GUESS 6560

/** Data Structure for a word */
typedef struct word{
    char text[8];
} word_t;

typedef int coloring_t;

/**
 * Word comparison function based on the wordle rules.
 * @param query The query word you would input into the wordle board
 * @param answer The underlying answer
 * @returns The coloring of the board. See the documentation below for details.
*/
coloring_t word_cmp(word_t &query, word_t &answer);

/**
 * Convert a cpp string type to the word type
 * @param source The input cpp string
 * @param dest the word_t data type to be written to
 * This function will always truncate the source string to WORDLEN size.
*/
void str2word(std::string &source, word_t &dest);

/**
 * Compare if two words are equal.
*/
bool word_eq(word_t &A, word_t &B);

/**
 * Deep Copy for the word data type
*/
void word_deepcopy(word_t &source, word_t &dest);

void word_print(word_t &word, coloring_t coloring = PTN_DEFAULT,
    char delim = '\n');

/**
 * Coloring is coded by a base-3 representation of an integer. 
 * The right most 3-digit represent the coloring of the 0th letter
 * Digit 0: Represents Grey in Wordle
 * Digit 1: Represents Yellow in Wordle
 * Digit 2: Represents Green in Wrodle
*/


#endif /* WORD_H */