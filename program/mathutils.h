/**
 * Header File for all math operations used in the wordle solver
 * #include "mathutils.h"
*/

#include <cmath>
#include <vector>
#include "word.h"

#ifndef MATH_H
#define MATH_H

#define PRECISION 1e-12
typedef coloring_t index_t;

/** Single term entropy computation */
float entropy(float prob);

/**
 * Generic Scatter Reduce Function
 * @param index An array of indices
 * @param in The input floating point array (in.size() == index.size())
 * @param out The output floating point array
 * @param multipler
*/
void scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out, float multiplier = 1.0);

/**
 * Masked Scatter Reduce Function
 * @param index An array of indices
 * @param in The input floating point array (in.size() == index.size())
 * @param out The output floating point array
 * @param mask
 * @param multipler
*/
void masked_scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out, std::vector<bool> &mask, float multiplier = 1.0);

/**
 * Generic map reduce with sum as the reduction function:
 * Returns \sum_{x \in vec} f(x)
*/
float map_reduce_sum(std::vector<float> vec, float (*f) (float));

#endif /* WORD_H */