/**
 * Header File for The main CUDA solver
*/

#ifndef CUDA_SOLVER_H
#define CUDA_SOLVER_H

#include "word.h"


/**
 * Class declaration inspired by the circle rendering program
*/
class CudaSolver {

private:

// Items accessible to CPU
size_t num_words;
float  prior_sum;
word_t *     _CPU_wordlist;
float *      _CPU_priorlist;

// Items accessible to GPU
word_t *     _CUDA_wordlist;
float *      _CUDA_priorlist;
coloring_t * _CUDA_pattern_matrix;
float *      _CUDA_word_scores;

// Temporary variables used by the GPU solver
// Needs to be visible across multiple kernel call 
float *     solver_prior_sum;
int *       solver_words_remaining;
int *       solver_guess_candidate;
float *     solver_candidate_score;
coloring_t *solver_feedback;

// Temporary device memory for scratch space. Supplied to 
// Reduce sum and Argmax
int *       _CUDA_scratch_int;
float *     _CUDA_scratch_float;


public:

CudaSolver();
virtual ~CudaSolver();

/**
 * Fill in the data fields of the solver. (Should only be called once)
 * @param word_list A list of allowed words
 * @param prior_list A list of prior weights
 * @param test_list A list of indicies of words to test
 * @param prior_sum The sum of all prior weights 
*/
void data_copy(std::vector<word_t> &word_list,
           std::vector<float> &prior_list, 
           float prior_sum_in);


/**
 * Initialize the cuda solver (Should only be called once)
 * @warning Please call data_copy before initializing the cuda solver.
*/
void cuda_setup();


/**
 * CUDA data parallel routine for pattern matrix computation
 * @warning This function must be called before running any solver routines
*/
void pattern_comute();

/**
 * Main solver routine for CUDA solver
 * @param test_idx The index to the answer word in the word list
 * @returns The number of game iterations used to solve the word
*/
int solve(int test_idx);


/**
 * Debugging routines: may take a while to run
*/
bool pattern_matrix_verify();

int solve_verbose(int test_idx);

};


#endif /*CUDA_SOLVER_H*/