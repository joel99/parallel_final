#include "wordle.h"
#include <iostream>
#include <cassert>

int main() {
    word_t test;
    std::string initialGuess = "steam"; // Create a std::string object from the string literal.
    str2word(initialGuess, test); // more common, salet is not in common list and is slower to iterate atm
    return 0;
}
