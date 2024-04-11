/**
 * Header File for Word Data Structure and Word wise Operations
 * #include "word.h"
*/

#include <iostream>

#define WORDLEN 8 // Maximum word length

#ifndef WORD_H
#define WORD_H

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
 * Debugging Function: Visualize the coloring. ("-": grey, "y": yellow, "g": green)
*/
void color_check(coloring_t coloring, word_t &query, word_t &answer);

/**
 * Coloring is coded by a base-3 representation of an integer. 
 * The right most 3-digit represent the coloring of the 0th letter
 * Digit 0: Represents Grey in Wordle
 * Digit 1: Represents Yellow in Wordle
 * Digit 2: Represents Green in Wrodle
*/


#endif /* WORD_H */