// Contains the solver logic.

#include "wordle.h"
#include <numeric>
#include <cassert>

std::vector<std::float_t> get_weights(const std::vector<word_t> &possibilities, const std::vector<std::float_t> &priors)
{
    /*
        possibilities: words with nonzero prob of being answer (based on feedback) TODO replace with mask
        priors: external prior for likelihood of being answer, should match size of possibilities

        return normalized weights
    */
    float total = 0;
    std::vector<std::float_t> weights;
    for (int i = 0; i < possibilities.size(); i++)
    {
        total += priors[i];
    }
    if (total == 0)
    {
        return std::vector<std::float_t>(possibilities.size(), 0.0);
    }
    for (int i = 0; i < possibilities.size(); i++)
    {
        weights.push_back(priors[i] / total);
    }
    return weights;
}

std::vector<std::float_t> get_entropies(
    const std::vector<word_t> &possibilities, 
    const std::vector<std::float_t> &weights,
    const std::vector<std::vector<coloring_t>> &coloring_matrix
)
{
    /*
        possibilities: words with nonzero prob of being answer (based on feedback)
        weights: normalized weights of possibilities, of length possibilities
        coloring_matrix: shape Guess x possibilities, elements 0-3^MAX_LETTERS

        Scatter reduce weights to different feedback labels for each guess.

        returns: vector of entropies of length Guess
    */
    if (std::accumulate(weights.begin(), weights.end(), 0.0f) == 0)
    {
        return std::vector<std::float_t>(coloring_matrix.size(), 0.0);
    }
    // Verify shapes
    assert(coloring_matrix[0].size() == weights.size() && "Coloring matrix must match weights.");
    assert(coloring_matrix[0].size() == possibilities.size() && "Coloring matrix must match possibilities.");
    // Initialize a 2D matrix of length Guess x 3^MAX_LETTERS
    std::vector<std::vector<std::float_t>> probs(coloring_matrix.size(), std::vector<std::float_t>(std::pow(3, MAX_LETTERS), 0.0f));

    // Serial implementation right now
    for (int guess = 0; guess < probs.size(); guess++)
    {
        for (int candidate = 0; candidate < coloring_matrix[guess].size(); candidate++)
        {
            coloring_t idx = coloring_matrix[guess][candidate];
            probs[guess][idx] += weights[candidate];
        }
    }
    
    // prob to entropy
    std::vector<std::float_t> entropies;
    for (int i = 0; i < probs.size(); i++)
    {
        // print start loop idx
        std::float_t entropy = calculate_entropy(probs[i]);
        entropies.push_back(entropy);
    }
    // print out of this loop
    return entropies;
}

word_t optimal_guess(
    const std::vector<word_t> &choices, // legal choices
    const std::vector<word_t> &possibilities,
    const std::vector<std::float_t> &priors, // TODO add the mask that reduces this
    const std::vector<std::vector<coloring_t>> &coloring_matrix // shape Guess x possibilities
)
{
    /*
        possibilities: TODO replace with indices into a color matrix, or color matrix directly
        priors: vector to scatter reduce
        ? Ummm need to pass the choices
        ! Currently this is just info max. Other strategies not implemented.
    */
    // Don't enable until we figure out why reduction is so dramatic
    if (possibilities.size() == 1) // TODO enable once ready to implement
    // if (true || possibilities.size() == 1) // TODO enable once ready to implement
    {
        return possibilities[0];
    }
    assert(possibilities.size() == priors.size() && "Priors must match possibilities");
    assert(possibilities.size() == coloring_matrix[0].size() && "Coloring matrix must match possibilities");

    std::vector<std::float_t> weights = get_weights(possibilities, priors); // this appears to be a normalizing step on priors.
    // std::cout << "Weights calculated." << std::endl;
    // std::cout << "Sample weights: " << weights[0] << ", " << weights[1] << ", " << weights[2] << std::endl;
    std::vector<std::float_t> ents = get_entropies(possibilities, weights, coloring_matrix); // scatter reduce
    // return choices[0];
    // std::cout << "Entropies calculated." << std::endl;

    return choices[std::distance(ents.begin(), std::max_element(ents.begin(), ents.end()))]; // argmax - this likely can be parallel.
}

word_t get_next_guess(
    const std::vector<word_t> &guesses,
    const std::vector<int> &patterns,
    const std::vector<word_t> &possibilities,
    const std::vector<word_t> &choices,
    const std::vector<std::float_t> &priors,
    const std::vector<std::vector<coloring_t>> &coloring_matrix
)
{
    /*
        possibilities: words with nonzero prob of being answer (based on feedback)
        choices: words that can be guessed (even if not in possibilities)
        priors: external prior for likelihood of being answer, should match size of possibilities

        3B1B's get_next_guess implements a 1 deep recursive tree search in `brute_force_optimal_guess`
        But generally forwards to `optimal_guess`, which is what is implemented here.

        3B1B also implements a next_guess_map to cache guesses given states of the board;
        - this seems useful but hard to think about in terms of parallelism, // TODO?
        We maintain the get_next_guess shell in case we want to add back in.
    */

    // TODO choices = get_possible_words(guess, pattern, choices)
    // Looks like 3B1B reduces choices in hard mode. For now we don't reduce at all, JY hasn't interpreted what this is for yet.
    return optimal_guess(choices, possibilities, priors, coloring_matrix);
}

std::vector<bool> get_possible_words_matrix(
    const int guess_idx, 
    coloring_t pattern, 
    const std::vector<std::vector<coloring_t>> &coloring_matrix)
{
    // return vector of length possibilities (coloring_matrix.size(1))
    // Index the matrix, simply.
    auto row = coloring_matrix[guess_idx]; // ? Is this a copy or a reference?
    std::vector<bool> mask(row.size(), false);
    for (int i = 0; i < row.size(); i++)
    {
        mask[i] = (row[i] == pattern);
    }
    return mask;
}