/**
 * Header File for all math operations used in the wordle solver
 * #include "mathutils.h"
*/

#include <cmath>
#include <vector>
#include <cstdlib>
#include "word.h"

#ifndef MATH_H
#define MATH_H

#define PRECISION 1e-12
typedef coloring_t index_t;

/**
 * Test if a floating point number is 0.
*/
bool is_zero(float x);

/**
 * returns a random floating point number between low and high.
*/
float f_rand(float low, float high);

/**
 * Generic Scatter Reduce Function
 * @param index An array of indices
 * @param in The input floating point array (in.size() == index.size())
 * @param out The output floating point array
 * @param multipler
*/
void scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out);

/**
 * Masked Scatter Reduce Function
 * @param index An array of indices
 * @param in The input floating point array (in.size() == index.size())
 * @param out The output floating point array
 * @param mask
 * @param multipler
*/
void masked_scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out, std::vector<bool> &mask);

/**
 * Parallel Scatter Reduce Function
 * @param index An array of indices
 * @param in The input floating point array (in.size() == index.size())
 * @param out The output floating point array
 * @param scratch A 2D array of size (n_proc, out.size())
*/
void parallel_scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out);
    // std::vector<float> &out, std::vector<std::vector<float>> &scratch);

/**
 * Computes the entropy via a map reduce operation
 * @param floats - Either a probability or a pooled weights
 * @param noramlize - The constant multiple applied to each term to normalize
 * into a probability distribution.
*/
float entropy_compute(std::vector<float> floats, float normalize = 1.0f);


/**
 * Computes the entropy via a map reduce operation
 * @param floats - Either a probability or a pooled weights
 * @param noramlize - The constant multiple applied to each term to normalize
 * into a probability distribution.
*/
void parallel_entropy_compute(std::vector<float> floats, float normalize, float& out);


#endif /* WORD_H */