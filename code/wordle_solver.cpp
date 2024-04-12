// Contains the solver logic.

#include "wordle.h"
#include <numeric>
#include <cassert>

coloring_t get_pattern(const std::string &guess, const std::string &answer)
{
    /*
        Returns coloring_t.
    */
    // return 0; // Assume 0 is a placeholder pattern
    // Indicator array, prevent matching the same letter twice. Init'd to false
    bool query_matched[MAX_LETTERS] = {0};
    bool answer_matched[MAX_LETTERS] = {0};

    coloring_t out = 0;
    coloring_t mult = 1;
    // Check for green boxes first
    for (int i = 0; i < MAX_LETTERS; i++)
    {
        if (guess[i] == answer[i])
        { // todo restore word_t struct
            out += (2 * mult);
            query_matched[i] = true;
            answer_matched[i] = true;
        }
        mult *= 3;
    }

    // reset multiplier
    mult = 1;
    // Check for yellow boxes
    for (int i = 0; i < MAX_LETTERS; i++)
    { // query index
        if (query_matched[i])
        {
            mult *= 3;
            continue;
        }
        for (int j = 0; j < MAX_LETTERS; j++)
        { // answer index
            if (i == j || answer_matched[j])
                continue;
            if (guess[i] == answer[j])
            {
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

std::vector<std::float_t> get_weights(const std::vector<std::string> &possibilities, const std::vector<std::float_t> &priors)
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
    const std::vector<std::string> &possibilities, 
    const std::vector<std::float_t> &weights,
    const std::vector<std::vector<coloring_t>> &coloring_matrix // assume this is of same length as possiblities
)
{
    /*
        possibilities: words with nonzero prob of being answer (based on feedback)
        weights: normalized weights
    */
    if (std::accumulate(weights.begin(), weights.end(), 0.0f) == 0)
    {
        return std::vector<std::float_t>(possibilities.size(), 0.0);
    }
    // Initialize a 2D matrix of length possibilites x 3^MAX_LETTERS
    std::vector<std::vector<std::float_t>> probs(possibilities.size(), std::vector<std::float_t>(std::pow(3, MAX_LETTERS), 0.0f));

    // Serial implementation right now
    for (int i = 0; i < probs.size(); i++)
    {
        for (int j = 0; j < probs[i].size(); j++)
        {
            coloring_t idx = coloring_matrix[i][j];
            probs[i][idx] += weights[i];
        }
    }
    
    // prob to entropy
    std::vector<std::float_t> entropies;
    for (int i = 0; i < probs.size(); i++)
    {
        entropies.push_back(calculate_entropy(probs[i]));
    }
}

std::string optimal_guess(
    const std::vector<std::string> &choices, // legal choices
    const std::vector<std::string> &possibilities,
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
    // hm... does this need to happen before or after?
    std::vector<std::float_t> ents = get_entropies(possibilities, weights, coloring_matrix); // scatter reduce
    // ents needs to be a vector of ints
    return choices[std::distance(ents.begin(), std::max_element(ents.begin(), ents.end()))]; // argmax - this likely can be parallel.
}

std::string get_next_guess(
    const std::vector<std::string> &guesses,
    const std::vector<int> &patterns,
    const std::vector<std::string> &possibilities,
    const std::vector<std::string> &choices,
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

std::vector<std::string> get_possible_words(const std::string &guess, coloring_t pattern, const std::vector<std::string> &possibilities)
{
    // Basic serial implementation
    std::vector<std::string> filteredWords;
    for (const auto &word : possibilities)
    {
        if (get_pattern(guess, word) == pattern)
        {
            filteredWords.push_back(word);
        }
    }
    return filteredWords;
}

std::vector<bool> get_possible_words_matrix(const int guess_idx, coloring_t pattern, const std::vector<std::vector<coloring_t>> &coloring_matrix)
{
    // Index the matrix, simply.
    auto row = coloring_matrix[guess_idx]; // ? Is this a copy or a reference?
    std::vector<bool> mask(row.size(), false);
    for (int i = 0; i < row.size(); i++)
    {
        mask[i] = (row[i] == pattern);
    }
    return mask;
}