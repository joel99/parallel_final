#include "word.h"
#include "utils.h"
#include "mathutils.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cstring>
#include <chrono>
#include <unistd.h>

// Items Specific to the MPI solver
#include <mpi.h>

#define MAXITERS 10

typedef struct MPI_program_args{
    size_t wordlist_size;
    size_t testset_size;
    float priors_sum;
} args_t;

#define COMM MPI_COMM_WORLD // I am tired of typing this over and over again.


void debug(int line_num){
  int pid;
  MPI_Comm_rank(COMM, &pid);
  printf("Process %d: %d\n", pid, line_num);
}

// Global Parameter: Maximum word length in use
int wordlen = MAXLEN;

// Macros for Timing Measurements
#define timestamp std::chrono::steady_clock::now() 
#define TIME(start, end) std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()

unsigned long ceil_xdivy(unsigned long X, unsigned long Y){
    return (X + (Y - 1)) / Y;
}

void usage(char *exec_name){
    int pid;
    MPI_Comm_rank(COMM, &pid);
    if(pid == 0){
    std::cout << "Usage:\n" << exec_name << " -f <word list> [-p <prior weights> -t <test list> -m <maximum word length> -r -v] \n";
    std::cout << "-v: verbose mode\n-r: use randomized priors";
    std::cout << "-m: specifies the maximum word length. Must be in between 1 and 8 (default)";
    std::cout << "-M: specifies the program Mode. 'D' - Default mode, 'O' - On the Fly Computation, 'P' - On the Fly Computation with Opener Preprocessing.";
    std::cout << "The test list must contain words in the word list\n";
    }

    return;
}

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

template <typename T, typename A>
int arg_min(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), min_element(vec.begin(), vec.end())));
}

/**
 * Determines the Candidate axis work region on the coloring matrix for each thread
 * @param total_size [in] The total number of words
 * @param num_procs [in] The number of processors
 * @param pid [in] Thread id
 * @param start [out] The starting index of the work region (inclusive)
 * @param end [out] The ending index of the work region (exclusive)
*/
void task_split(int total_size, int num_procs, int pid,
    int &start, int &end){
    int avg_count = total_size / num_procs;
    int remainder = total_size % num_procs;
    int task_size = (pid < remainder) ? avg_count + 1: avg_count;
    start = (pid < remainder) ? (avg_count + 1) * pid : (avg_count * pid) + remainder;
    end = start + task_size;
}

/**
 * Get the pid, given a global index. Pid inferred from task split.
*/
int get_pid_from_index(int total_size, int num_procs, int index){
    int avg_count = total_size / num_procs;
    int remainder = total_size % num_procs;
    int guess = 0;
    int lo = 0, hi = 0, task_size;
    while(lo < total_size){
        task_size = (guess < remainder) ? avg_count + 1: avg_count;
        hi = lo + task_size;
        if(index >= lo && index < hi) return guess;
        guess += 1;
        lo = hi;
    }
    return -1; // Error.
}



/**
 * This function computes the entirety of the coloring pattern matrix
 * using pair-wise word comparison. This is a massive data parallel procedure
 * @param pattern_matrix The pattern matrix to be written to
 * @param words The list of words
*/
void compute_patterns(std::vector<std::vector<coloring_t>> &pattern_matrix,
                      wordlist_t &words){
    int num_words = words.size();
    for(int query_idx = 0; query_idx < num_words; query_idx++){
        word_t query = words[query_idx];
        for (int candidate_idx = 0; candidate_idx < num_words; candidate_idx++){
            pattern_matrix[query_idx][candidate_idx] = 
                word_cmp(query, words[candidate_idx]);
        }
    }
}

/**
 * This function computes the entirety of the coloring pattern matrix
 * using pair-wise word comparison. This is a massive data parallel procedure
 * @param pattern_matrix The pattern matrix to be written to
 * @param words The list of words
 * @param work_start The starting index of a thread's work region
 * @param work_end The ending index of a thread's work region (exclusive)
*/
void MPI_compute_patterns(std::vector<std::vector<coloring_t>> &pattern_matrix,
                          wordlist_t &words,
                          int work_start,
                          int work_end){
    int num_words = words.size();     
    for(int query_idx = work_start; query_idx < work_end; query_idx++){
        word_t query = words[query_idx];
        for (int candidate_idx = 0; candidate_idx < num_words; candidate_idx++){
            // addressing in the thread's local address space
            pattern_matrix[query_idx - work_start][candidate_idx] = 
                word_cmp(query, words[candidate_idx]);
        }
    }
}

/**
 * Verbose Mode Solver for MPI: Requires word list for information output.
 * @param prior - The prior weights of each word. 
 * @param pattern_matrix the coloring pattern matrrx.
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @param work_start - The start index of a thread's work region
 * @param work_end - The end index of a thread's work region
 * @warning This function destructively modifies the priors vector.
*/
int solver_verbose(wordlist_t &words,
            priors_t &priors,
            std::vector<std::vector<coloring_t>> &pattern_matrix,
            int &answer,
            float prior_sum,
            int work_start,
            int work_end){
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    // Initialize Additional Solver Data
    int num_words = priors.size();
    int words_remaining = num_words;
    int num_patterns = get_num_patterns();
    // Scratch work and entrypy storage.
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_end - work_start, 0.0f);

    int guess; // The index (Relative to the word list) of each thread's guess word
    float guess_score; // The score of each thread's guessed word
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 
    // Buffer to store communication results
    int selection_root;

    coloring_t feedback;
    // Computes the initial uncertainty measure
    float uncertainty = entropy_compute(priors, prior_sum);

    if(pid == 0)
        std::cout << "Initial Uncertainty: " << uncertainty << "\n";
    bool random_select;

    MPI_Barrier(COMM);

    int iters = 0;

    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        if(pid == 0)
            std::cout<<"==========================================================\n";
        random_select = false;
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(priors);
            guess_score = priors[guess];
            random_select = true;
            // DEBUG CODE:
            int rank = 0;
            while (rank < num_proc) {
                if (pid == rank) {
                    std::cout << "pid :" << pid << ": " << work_start << "," << work_end << "\n";
                    // std::cout << "Scatter Reduce + Entropy Computation Time: " << TIME(compute_start, compute_end) << "\n";
                    // std::cout << "Word Selection Time:" << TIME(compute_end, select_end) << "\n";
                    std::cout << "Selected Word: " << std::flush;
                    word_print(words[guess], 0, ' ');
                    std::cout << "Score: " << guess_score << "\n" << std::flush;
                }
                rank ++;
                MPI_Barrier(COMM);
            }
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            auto compute_start = timestamp;
            for(int word_idx = work_start; word_idx < work_end; word_idx++){
                probability_scratch.assign(num_patterns, 0.0f);
                // Pool up the total word weights for each pattern
                scatter_reduce(pattern_matrix[word_idx - work_start], priors,
                    probability_scratch);
                // Normalize the pooled weights into a probability distribution
                // Store the score of this work in the local vector
                entropys[word_idx - work_start] = entropy_compute(probability_scratch, 
                    prior_sum)+ (priors[word_idx] / prior_sum);
            }
            auto compute_end = timestamp;
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            guess_score = entropys[guess];
            guess += work_start; // Convert local index to global index.
            auto select_end = timestamp;

            // DEBUG CODE:
            int rank = 0;
            while (rank < num_proc) {
                if (pid == rank) {
                    std::cout << "pid :" << pid << ": " << work_start << "," << work_end << "\n";
                    // std::cout << "Scatter Reduce + Entropy Computation Time: " << TIME(compute_start, compute_end) << "\n";
                    // std::cout << "Word Selection Time:" << TIME(compute_end, select_end) << "\n";
                    std::cout << "Selected Word: " << std::flush;
                    word_print(words[guess], 0, ' ');
                    std::cout << "Score: " << guess_score << "\n" << std::flush;
                }
                rank ++;
                MPI_Barrier(COMM);
            }
        }
        if(random_select){
            // If a word is selected randomly, then no need to communicate.
            selection_root = get_pid_from_index(num_words, num_proc, guess);
        }
        else{
            // Communicate Each thread's local result
            MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
                (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
            MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
                (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
            // Select Word for all threads. selection_root is the thread
            // that contributed the optimal guess
            selection_root = arg_max(all_scores);
            guess = all_guesses[selection_root];
        }

        

        // DEBUG CODE:
        if(pid == 0){
            std::cout << "Word:";
            word_print(words[guess], 0, ' ');
            std::cout << "is selected. (" << selection_root << ") with ";
            std::cout << "Score: " << all_scores[selection_root] << "\n" << std::flush;
        }

        /******************** Update Phase **********************/
        MPI_Barrier(COMM); // All threads must agree upon which thread has the guessed word.
        auto update_start = timestamp;
        if(pid == selection_root){
            // Obtain feedback in the local address space
            feedback = pattern_matrix[guess - work_start][answer];
            words_remaining = 0;
            prior_sum = 0.0f;
            for(int i = 0; i < num_words; i++){
                if(is_zero(priors[i])) continue; // prior == 0 for invalid
                if(pattern_matrix[guess - work_start][i] != feedback) 
                    priors[i] = 0.0f;
                else{
                    words_remaining += 1;
                    prior_sum += priors[i];
                }
            }
        }
        // The selection root broadcasts the updated prior list to all threads
        MPI_Bcast((void *) &feedback, sizeof(coloring_t), MPI_BYTE, selection_root, COMM);
        MPI_Bcast((void *) &words_remaining, sizeof(int), MPI_BYTE, selection_root, COMM);
        MPI_Bcast((void *) &prior_sum, sizeof(float), MPI_BYTE, selection_root, COMM);
        MPI_Bcast((void *) &(priors.front()),
            sizeof(float) * num_words, MPI_BYTE, selection_root, COMM);

        // Compute the new uncertainty measure after a guess
        uncertainty = entropy_compute(priors, prior_sum);
        auto update_end = timestamp;

        // DEBUG CODE:
        int rank = 0;
        while (rank < num_proc) {
            if (pid == rank) {
                std::cout << "post update check: pid :" << pid << ": " << work_start << "," << work_end << "\n";
                // std::cout << "Update Phase total Time:" << TIME(update_start, update_end) << "\n"; 
                std::cout << "Check if Feedback Agrees:";
                word_print(words[guess], feedback);
                std::cout << "Check: prior_sum = " << prior_sum << " num_remain = " << words_remaining; 
                // for(int i = 0; i < num_words; i++){
                //     if(!is_zero(priors[i])) word_print(words[i], 0, ' ');
                // }
                std::cout << "\n==============================================\n" << std::flush;
            }
            rank ++;
            MPI_Barrier(COMM);
        }
        iters ++;

        // All threads should escape the loop if the guess is correct.
        if(is_correct_guess(feedback)) return iters;
    }
    return iters;
}


/**
 * Main Solver for MPI: Eliminated the need to input the word list.
 * @param prior - The prior weights of each word. 
 * @param pattern_matrix the coloring pattern matrrx.
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @param work_start - The start index of a thread's work region
 * @param work_end - The end index of a thread's work region
 * @warning This function destructively modifies the priors vector.
*/
int solver(priors_t &priors,
            std::vector<std::vector<coloring_t>> &pattern_matrix,
            int &answer,
            float prior_sum,
            int work_start,
            int work_end){
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    // Initialize Additional Solver Data
    int num_words = priors.size();
    int words_remaining = num_words;
    int num_patterns = get_num_patterns();
    // Scratch work and entrypy storage.
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_end - work_start, 0.0f);

    int guess; // The index (Relative to the word list) of each thread's guess word
    float guess_score; // The score of each thread's guessed word
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 
    // Buffer to store communication results
    int selection_root;
    bool random_select;

    coloring_t feedback;

    int iters = 0;

    while(iters < MAXITERS){
        random_select = false;
        /******************** Entropy Computation Phase **********************/
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(priors);
            guess_score = priors[guess];
            random_select = true;
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            for(int word_idx = work_start; word_idx < work_end; word_idx++){
                probability_scratch.assign(num_patterns, 0.0f);
                // Pool up the total word weights for each pattern
                scatter_reduce(pattern_matrix[word_idx - work_start], priors,
                    probability_scratch);
                // Normalize the pooled weights into a probability distribution
                // Store the score of this work in the local vector
                entropys[word_idx - work_start] = entropy_compute(probability_scratch, 
                    prior_sum)+ (priors[word_idx] / prior_sum);
            }
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            guess_score = entropys[guess];
            guess += work_start; // Convert local index to global index.
        }

        if(random_select){
            // If a word is selected randomly, then no need to communicate.
            selection_root = get_pid_from_index(num_words, num_proc, guess);
        }
        else{
            // Communicate Each thread's local result
            MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
                (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
            MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
                (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
            // Select Word for all threads. selection_root is the thread
            // that contributed the optimal guess
            selection_root = arg_max(all_scores);
            guess = all_guesses[selection_root];
        }


        /******************** Update Phase **********************/
        MPI_Barrier(COMM); // All threads must agree upon which thread has the guessed word.
        if(pid == selection_root){
            // Obtain feedback in the local address space
            feedback = pattern_matrix[guess - work_start][answer];
            words_remaining = 0;
            prior_sum = 0.0f;
            for(int i = 0; i < num_words; i++){
                if(is_zero(priors[i])) continue; // prior == 0 for invalid
                if(pattern_matrix[guess - work_start][i] != feedback) 
                    priors[i] = 0.0f;
                else{
                    words_remaining += 1;
                    prior_sum += priors[i];
                }
            }
        }
        // The selection root broadcasts the updated prior list to all threads
        MPI_Bcast((void *) &feedback, sizeof(coloring_t), MPI_BYTE, selection_root, COMM);
        MPI_Bcast((void *) &words_remaining, sizeof(int), MPI_BYTE, selection_root, COMM);
        MPI_Bcast((void *) &prior_sum, sizeof(float), MPI_BYTE, selection_root, COMM);
        MPI_Bcast((void *) &(priors.front()),
            sizeof(float) * num_words, MPI_BYTE, selection_root, COMM);

        iters ++;
        // All threads should escape the loop if the guess is correct.
        if(is_correct_guess(feedback)) return iters;
    }
    return iters;
}

/************************************
 * Solver with No Precomputation
*************************************/

/**
 * Verbose Mode Solver for MPI: Requires word list. 
 * Computes the pattern on the fly and does not use the pattern matrix as the
 * input. This implementation also implicitly incorporates rebuilding strategies
 * 
 * @param words - The list of words, required for this implementation
 * @param priors - The prior weights of each word. 
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @warning This function makes its own copy of prior vectors
*/
int solver_verbose_no_precompute(wordlist_t &words,
            priors_t &priors,
            int answer,
            float prior_sum){
    // Obtain the Pid and computes the word region of each thread.
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    int work_start, work_end, work_size;
    task_split(static_cast<int>(words.size()), num_proc, pid, work_start, work_end);
    work_size = work_end - work_start;


    // Initialize Constants
    int num_words = words.size();
    int words_remaining = num_words;
    int num_patterns = get_num_patterns();
    word_t answer_word = words[answer];

    // Initialize the prior vectors to be destructively modified by the solver
    std::vector<float> candidate_priors(priors); // Statically sized
    std::vector<float> answer_priors(priors);    // Dynamically sized
    std::vector<int> indecies(num_words, 0);
    for (int i = 0; i < num_words; i++){
        indecies[i] = i;
    } // Initialize an array of word indecies: Necessary for dynamic rebuilding


    // Scratch work and entrypy storage. Private to each thread
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_size, 0.0f);

    
    // MPI Communication Buffers and data:
    int guess; // The index (Relative to the word list) of each thread's guess word
    float guess_score; // The score of each thread's guessed word
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 
    int selection_root;

    coloring_t feedback;

    // DEBUGGING CODE: Computes the initial uncertainty
    float uncertainty = entropy_compute(priors, prior_sum);
    if(pid == 0)
        std::cout << "Initial Uncertainty: " << uncertainty << "\n";

    bool random_select;

    MPI_Barrier(COMM);

    int iters = 0;

    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        if(pid == 0)
            std::cout<<"==========================================================\n";
        random_select = false;
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(candidate_priors);
            guess_score = candidate_priors[guess];
            random_select = true;
            // DEBUG CODE:
            int rank = 0;
            while (rank < num_proc) {
                if (pid == rank) {
                    std::cout << "pid :" << pid << ": " << work_start << "," << work_end << "\n";
                    std::cout << "Selected Word: " << std::flush;
                    word_print(words[guess], 0, ' ');
                    std::cout << "Score: " << guess_score << "\n" << std::flush;
                }
                rank ++;
                MPI_Barrier(COMM);
            }
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            auto compute_start = timestamp;
            // Use thread local indexing
            for(int lcl_idx = 0; lcl_idx < work_size; lcl_idx++){
                // Zero out the probability scratch vector for new candidate
                probability_scratch.assign(num_patterns, 0.0f);

                // Computes the global word List index:
                int candidate_idx = lcl_idx + work_start;
                word_t candidate = words[candidate_idx];
                // Compute the coloring probability scratch vector on the fly:
                for(int j = 0; j < words_remaining; j++){
                    int answer_idx = indecies[j];
                    int weight = answer_priors[j];
                    coloring_t tmp_feedback = word_cmp(candidate, words[answer_idx]);
                    // Pool the prior weights into the probability scratch
                    probability_scratch[static_cast<int>(tmp_feedback)] += weight;
                }
                // Normalize the pooled weights, then compute the entropy score
                entropys[lcl_idx] = entropy_compute(probability_scratch, 
                    prior_sum) + (candidate_priors[candidate_idx] / prior_sum);
            }
            auto compute_end = timestamp;
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            guess_score = entropys[guess];
            guess += work_start; // Convert local index to global index.
            auto select_end = timestamp;

            // DEBUG CODE:
            int rank = 0;
            // while (rank < num_proc) {
                if (pid == 0) {
                    std::cout << "pid :" << pid << ": " << work_start << "," << work_end << "\n";
                    std::cout << "Entropy Computation Time: " << TIME(compute_start, compute_end) << "\n";
                    std::cout << "Word Selection Time:" << TIME(compute_end, select_end) << "\n";
                    std::cout << "Selected Word: " << std::flush;
                    word_print(words[guess], 0, ' ');
                    std::cout << "Score: " << guess_score << "\n" << std::flush;
                }
                rank ++;
                MPI_Barrier(COMM);
            // }
        }
        if(random_select){
            // If a word is selected randomly, then no need to communicate.
            selection_root = get_pid_from_index(num_words, num_proc, guess);
        }
        else{
            // Communicate Each thread's local result
            MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
                (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
            MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
                (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
            // Select Word for all threads. selection_root is the thread
            // that contributed the optimal guess
            selection_root = arg_max(all_scores);
            guess = all_guesses[selection_root];
        }

        



        /******************** Update Phase **********************/
        MPI_Barrier(COMM); // All threads must agree upon which thread has the guessed word.
        auto update_start = timestamp;
        // Instead of serializing the update phase and communicate the result,
        // Each thread could just do the same set of redundant operations

        // Compute the feedback of the guessed word:
        word_t guess_word = words[guess];
        prior_sum = 0.0f;
        feedback = word_cmp(guess_word, answer_word);
        int write_idx = 0; // Used to push the remaining valid words to the front

                // DEBUG CODE:
        if(pid == 0){

            std::cout << "Word:";
            word_print(words[guess], feedback, ' ');
            word_print(answer_word, feedback, ' ');
            std::cout << "is selected. (" << selection_root << ") with ";
            std::cout << "Score: " << all_scores[selection_root] << "\n" << std::flush;
        }

        /**
         * For the candidate_priors: simply zero out the invalid words
         * For the answer_priors: push the prior weights of valid words
         *      to the front of the array
         * For the index vector: Do similar operations as the answer priors list
        */
        for(int j = 0; j < words_remaining; j++){ 
            // Only consider the words that are left:
            int remain_idx = indecies[j];
            coloring_t tmp_feedback = word_cmp(guess_word, words[remain_idx]);
            if(tmp_feedback == feedback){ // Underlying word is consistent with the guess
                prior_sum += answer_priors[j];
                answer_priors[write_idx] = answer_priors[j];
                indecies[write_idx] = indecies[j];
                write_idx += 1;
            }
            else{// Word is eliminated
                candidate_priors[indecies[j]] = 0.0f;
            }
        }
        words_remaining = write_idx; // Update the number of words remaining


        // Compute the new uncertainty measure after a guess
        uncertainty = entropy_compute(answer_priors, prior_sum);
        auto update_end = timestamp;

        // DEBUG CODE:
        int rank = 0;
        // while (rank < num_proc) {
            if (pid == 0) {
                std::cout << "post update check: pid :" << pid << ": " << work_start << "," << work_end << "\n";
                std::cout << "Update Phase total Time:" << TIME(update_start, update_end) << "\n"; 
                std::cout << "Check if Feedback Agrees:";
                word_print(words[guess], feedback);
                std::cout << "Check: prior_sum = " << prior_sum << " num_remain = " << words_remaining << "Valid Words:\n"; 
                for(int i = 0; i < words_remaining; i++){
                    word_print(words[indecies[i]], 0, ' ');
                }
                std::cout << "\n==============================================\n" << std::flush;
            }
            rank ++;
            MPI_Barrier(COMM);
        // }
        iters ++;

        // All threads should escape the loop if the guess is correct.
        if(is_correct_guess(feedback)) return iters;

        // Barrier at the end of the iteration.
        MPI_Barrier(COMM);
    }
    return iters;
}


/**
 * Solver for MPI no precompute: Requires word list. 
 * Computes the pattern on the fly and does not use the pattern matrix as the
 * input. This implementation also implicitly incorporates rebuilding strategies
 * 
 * @param words - The list of words, required for this implementation
 * @param priors - The prior weights of each word. 
 * @param prior_sum - The sum of all prior weights, returned by the function
 *                    that generates the vector of prior weights
 * @param answer - The WORD INDEX of the correct word.
 * @warning This function makes its own copy of prior vectors
*/
int solver_no_precompute(wordlist_t &words,
            priors_t &priors,
            int answer,
            float prior_sum){
    // Obtain the Pid and computes the word region of each thread.
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    int work_start, work_end, work_size;
    task_split(static_cast<int>(words.size()), num_proc, pid, work_start, work_end);
    work_size = work_end - work_start;


    // Initialize Constants
    int num_words = words.size();
    int words_remaining = num_words;
    int num_patterns = get_num_patterns();
    word_t answer_word = words[answer];

    // Initialize the prior vectors to be destructively modified by the solver
    std::vector<float> candidate_priors(priors); // Statically sized
    std::vector<float> answer_priors(priors);    // Dynamically sized
    std::vector<int> indecies(num_words, 0);
    for (int i = 0; i < num_words; i++){
        indecies[i] = i;
    } // Initialize an array of word indecies: Necessary for dynamic rebuilding


    // Scratch work and entrypy storage. Private to each thread
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_size, 0.0f);

    
    // MPI Communication Buffers and data:
    int guess; // The index (Relative to the word list) of each thread's guess word
    float guess_score; // The score of each thread's guessed word
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 

    coloring_t feedback;

    MPI_Barrier(COMM);

    int iters = 0;

    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(candidate_priors);
            guess_score = candidate_priors[guess];
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            // Use thread local indexing
            for(int lcl_idx = 0; lcl_idx < work_size; lcl_idx++){
                // Zero out the probability scratch vector for new candidate
                probability_scratch.assign(num_patterns, 0.0f);

                // Computes the global word List index:
                int candidate_idx = lcl_idx + work_start;
                word_t candidate = words[candidate_idx];
                // Compute the coloring probability scratch vector on the fly:
                for(int j = 0; j < words_remaining; j++){
                    int answer_idx = indecies[j];
                    int weight = answer_priors[j];
                    coloring_t tmp_feedback = word_cmp(candidate, words[answer_idx]);
                    // Pool the prior weights into the probability scratch
                    probability_scratch[static_cast<int>(tmp_feedback)] += weight;
                }
                // Normalize the pooled weights, then compute the entropy score
                entropys[lcl_idx] = entropy_compute(probability_scratch, 
                    prior_sum) + (candidate_priors[candidate_idx] / prior_sum);
            }
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            guess_score = entropys[guess];
            guess += work_start; // Convert local index to global index.

        }

        // Communicate Each thread's local result
        MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
            (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
        MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
            (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
        // Select Word for all threads.
        guess = all_guesses[arg_max(all_scores)];

        

        /******************** Update Phase **********************/
        MPI_Barrier(COMM); // All threads must agree upon which thread has the guessed word.
        // Instead of serializing the update phase and communicate the result,
        // Each thread could just do the same set of redundant operations

        // Compute the feedback of the guessed word:
        word_t guess_word = words[guess];
        prior_sum = 0.0f;
        feedback = word_cmp(guess_word, answer_word);
        int write_idx = 0; // Used to push the remaining valid words to the front

        /**
         * For the candidate_priors: simply zero out the invalid words
         * For the answer_priors: push the prior weights of valid words
         *      to the front of the array
         * For the index vector: Do similar operations as the answer priors list
        */
        for(int j = 0; j < words_remaining; j++){ 
            // Only consider the words that are left:
            int remain_idx = indecies[j];
            coloring_t tmp_feedback = word_cmp(guess_word, words[remain_idx]);
            if(tmp_feedback == feedback){ // Underlying word is consistent with the guess
                prior_sum += answer_priors[j];
                answer_priors[write_idx] = answer_priors[j];
                indecies[write_idx] = indecies[j];
                write_idx += 1;
            }
            else{// Word is eliminated
                candidate_priors[indecies[j]] = 0.0f;
            }
        }
        words_remaining = write_idx; // Update the number of words remaining

        iters ++;

        // All threads should escape the loop if the guess is correct.
        if(is_correct_guess(feedback)) return iters;

        // Barrier at the end of the iteration.
        MPI_Barrier(COMM);
    }
    return iters;
}


/************************************************
 * On The Fly solvers with Opener Precomputation
************************************************/

/**
 * For on the fly solvers: Precompute the opener word once
 * @param words The word list
 * @param priors The prior list
 * @param prior_sum: The sum of all priors weights
 * @return the index of the opener word in the word list
*/
int opener_precompute(wordlist_t &words, priors_t &priors, float prior_sum){
    // Obtain the Pid and computes the word region of each thread.
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    int work_start, work_end, work_size;
    task_split(static_cast<int>(words.size()), num_proc, pid, work_start, work_end);
    work_size = work_end - work_start;

    int num_words = words.size();
    int num_patterns = get_num_patterns();

    // Scratch work and entrypy storage. Private to each thread
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_size, 0.0f);

    // Communication Buffers
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 

    // Use thread local indexing
    for(int lcl_idx = 0; lcl_idx < work_size; lcl_idx++){
        // Zero out the probability scratch vector for new candidate
        probability_scratch.assign(num_patterns, 0.0f);
        // Computes the global word List index:
        int candidate_idx = lcl_idx + work_start;
        word_t candidate = words[candidate_idx];
        // Compute the coloring probability scratch vector on the fly:
        for(int j = 0; j < num_words; j++){
            coloring_t feedback = word_cmp(candidate, words[j]);
            // Pool the prior weights into the probability scratch
            probability_scratch[static_cast<int>(feedback)] += priors[j];
        }
        // Normalize the pooled weights, then compute the entropy score
        entropys[lcl_idx] = entropy_compute(probability_scratch, 
            prior_sum) + (priors[candidate_idx] / prior_sum);
    }

    // Find the word that maximizes the expected entropy entropy.
    int guess = arg_max(entropys);
    float guess_score = entropys[guess];
    guess += work_start; // Global word index
    // Communication phase: (Sync)
    MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
        (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
    MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
        (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
    // Return the index of the opener word maximizing the score
    return all_guesses[arg_max(all_scores)];
}

/**
 * Verbose Mode On The Fly Solver for MPI: Requires word list. 
 * Computes the pattern on the fly and does not use the pattern matrix as the
 * input. This implementation also implicitly incorporates rebuilding strategies
 * The opener word shall be pre-computed
 * 
 * @param words - The list of words, required for this implementation
 * @param priors - The prior weights of each word. 
 * @param answer - The WORD INDEX of the correct word.
 * @param opener - The WORD INDEX of the opener word
 * @warning This function makes its own copy of prior vectors
*/
int solver_OTF_opener_verbose(wordlist_t &words,
            priors_t &priors,
            int answer,
            int opener){
    // Obtain the Pid and computes the word region of each thread.
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    int work_start, work_end, work_size;
    task_split(static_cast<int>(words.size()), num_proc, pid, work_start, work_end);
    work_size = work_end - work_start;

    auto init_start = timestamp;

    // Initialize Constants
    float prior_sum = 0.0f;
    int num_words = words.size();
    int num_patterns = get_num_patterns();
    word_t answer_word = words[answer];
    word_t opener_word = words[opener];

    // Initialize the prior vectors to be destructively modified by the solver
    std::vector<float> candidate_priors(num_words, 0.0f); // Statically sized
    std::vector<float> answer_priors(num_words, 0.0f);    // Dynamically sized
    std::vector<int> indecies(num_words, 0);

    // Perform the first round of update according to the opener word:
    // Performed by all threads as a means to initialize the candidate and 
    // answer priors
    coloring_t feedback = word_cmp(opener_word, answer_word);
    int words_remaining = 0;
    for(int j = 0; j < num_words; j++){ 
        if(word_cmp(opener_word, words[j]) == feedback){ // Underlying word is consistent with the guess
            prior_sum += priors[j];
            candidate_priors[j] = priors[j];
            answer_priors[words_remaining] = priors[j];
            indecies[words_remaining] = j; // Keep track of the original indecies
            words_remaining += 1;
        }
    }
    // The following should be unchanged from the functions above


    // Scratch work and entrypy storage. Private to each thread
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_size, 0.0f);

    
    // MPI Communication Buffers and data:
    int guess; // The index (Relative to the word list) of each thread's guess word
    float guess_score; // The score of each thread's guessed word
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 
    int selection_root;

    // DEBUGGING CODE: Computes the initial uncertainty
    float uncertainty = entropy_compute(priors, prior_sum);
    if(pid == 0)
        std::cout << "Initial Uncertainty: " << uncertainty << "\n";

    auto init_end = timestamp;
    if(pid == 0){
        std::cout << "Init Time:" << TIME(init_start, init_end) << "\n"; 
        std::cout << "prior_sum = " << prior_sum << " num_remain = " << words_remaining << "Valid Words:\n"; 
        for(int i = 0; i < words_remaining; i++){
            word_print(words[indecies[i]], 0, ' ');
        }
        std::cout << "\n" << std::flush;
    }

    bool random_select;

    MPI_Barrier(COMM);

    int iters = 1; // One round of game has already passed.

    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        if(pid == 0)
            std::cout<<"==========================================================\n";
        random_select = false;
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(candidate_priors);
            guess_score = candidate_priors[guess];
            random_select = true;
            // DEBUG CODE:
            int rank = 0;
            while (rank < num_proc) {
                if (pid == rank) {
                    std::cout << "pid :" << pid << ": " << work_start << "," << work_end << "\n";
                    std::cout << "Selected Word: " << std::flush;
                    word_print(words[guess], 0, ' ');
                    std::cout << "Score: " << guess_score << "\n" << std::flush;
                }
                rank ++;
                MPI_Barrier(COMM);
            }
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            auto compute_start = timestamp;
            // Use thread local indexing
            for(int lcl_idx = 0; lcl_idx < work_size; lcl_idx++){
                // Zero out the probability scratch vector for new candidate
                probability_scratch.assign(num_patterns, 0.0f);

                // Computes the global word List index:
                int candidate_idx = lcl_idx + work_start;
                word_t candidate = words[candidate_idx];
                // Compute the coloring probability scratch vector on the fly:
                for(int j = 0; j < words_remaining; j++){
                    int answer_idx = indecies[j];
                    int weight = answer_priors[j];
                    coloring_t tmp_feedback = word_cmp(candidate, words[answer_idx]);
                    // Pool the prior weights into the probability scratch
                    probability_scratch[static_cast<int>(tmp_feedback)] += weight;
                }
                // Normalize the pooled weights, then compute the entropy score
                entropys[lcl_idx] = entropy_compute(probability_scratch, 
                    prior_sum) + (candidate_priors[candidate_idx] / prior_sum);
            }
            auto compute_end = timestamp;
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            guess_score = entropys[guess];
            guess += work_start; // Convert local index to global index.
            auto select_end = timestamp;

            // DEBUG CODE:
            int rank = 0;
            // while (rank < num_proc) {
                if (pid == 0) {
                    std::cout << "pid :" << pid << ": " << work_start << "," << work_end << "\n";
                    std::cout << "Entropy Computation Time: " << TIME(compute_start, compute_end) << "\n";
                    std::cout << "Word Selection Time:" << TIME(compute_end, select_end) << "\n";
                    std::cout << "Selected Word: " << std::flush;
                    word_print(words[guess], 0, ' ');
                    std::cout << "Score: " << guess_score << "\n" << std::flush;
                }
                rank ++;
                MPI_Barrier(COMM);
            // }
        }
        if(random_select){
            // If a word is selected randomly, then no need to communicate.
            selection_root = get_pid_from_index(num_words, num_proc, guess);
        }
        else{
            // Communicate Each thread's local result
            MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
                (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
            MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
                (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
            // Select Word for all threads. selection_root is the thread
            // that contributed the optimal guess
            selection_root = arg_max(all_scores);
            guess = all_guesses[selection_root];
        }

        // DEBUG CODE:
        if(pid == 0){
            std::cout << "Word:";
            word_print(words[guess], 0, ' ');
            std::cout << "is selected. (" << selection_root << ") with ";
            std::cout << "Score: " << all_scores[selection_root] << "\n" << std::flush;
        }

        /******************** Update Phase **********************/
        MPI_Barrier(COMM); // All threads must agree upon which thread has the guessed word.
        auto update_start = timestamp;
        // Instead of serializing the update phase and communicate the result,
        // Each thread could just do the same set of redundant operations

        // Compute the feedback of the guessed word:
        word_t guess_word = words[guess];
        prior_sum = 0.0f;
        feedback = word_cmp(guess_word, answer_word);
        int write_idx = 0; // Used to push the remaining valid words to the front

        /**
         * For the candidate_priors: simply zero out the invalid words
         * For the answer_priors: push the prior weights of valid words
         *      to the front of the array
         * For the index vector: Do similar operations as the answer priors list
        */
        for(int j = 0; j < words_remaining; j++){ 
            // Only consider the words that are left:
            int remain_idx = indecies[j];
            coloring_t tmp_feedback = word_cmp(guess_word, words[remain_idx]);
            if(tmp_feedback == feedback){ // Underlying word is consistent with the guess
                prior_sum += answer_priors[j];
                answer_priors[write_idx] = answer_priors[j];
                indecies[write_idx] = indecies[j];
                write_idx += 1;
            }
            else{// Word is eliminated
                candidate_priors[indecies[j]] = 0.0f;
            }
        }
        words_remaining = write_idx; // Update the number of words remaining


        // Compute the new uncertainty measure after a guess
        uncertainty = entropy_compute(answer_priors, prior_sum);
        auto update_end = timestamp;

        // DEBUG CODE:
        int rank = 0;
        // while (rank < num_proc) {
            if (pid == 0) {
                std::cout << "post update check: pid :" << pid << ": " << work_start << "," << work_end << "\n";
                std::cout << "Update Phase total Time:" << TIME(update_start, update_end) << "\n"; 
                std::cout << "Check if Feedback Agrees:";
                word_print(words[guess], feedback);
                std::cout << "Check: prior_sum = " << prior_sum << " num_remain = " << words_remaining << "Valid Words:\n"; 
                for(int i = 0; i < words_remaining; i++){
                    word_print(words[indecies[i]], 0, ' ');
                }
                std::cout << "\n==============================================\n" << std::flush;
            }
            rank ++;
            MPI_Barrier(COMM);
        // }

        iters ++;
        // All threads should escape the loop if the guess is correct.
        if(is_correct_guess(feedback)) return iters;

        // Barrier at the end of the iteration.
        MPI_Barrier(COMM);
    }
    return iters;
}

/**
 * On The Fly Solver for MPI: Requires word list. 
 * Computes the pattern on the fly and does not use the pattern matrix as the
 * input. This implementation also implicitly incorporates rebuilding strategies
 * The opener word shall be pre-computed
 * 
 * @param words - The list of words, required for this implementation
 * @param priors - The prior weights of each word. 
 * @param answer - The WORD INDEX of the correct word.
 * @param opener - The WORD INDEX of the opener word
 * @warning This function makes its own copy of prior vectors
*/
int solver_OTF_opener(wordlist_t &words,
            priors_t &priors,
            int answer,
            int opener){
    // Obtain the Pid and computes the word region of each thread.
    int pid;
    MPI_Comm_rank(COMM, &pid);
    int num_proc;
    MPI_Comm_size(COMM, &num_proc);

    int work_start, work_end, work_size;
    task_split(static_cast<int>(words.size()), num_proc, pid, work_start, work_end);
    work_size = work_end - work_start;

    // Initialize Constants
    float prior_sum = 0.0f;
    int num_words = words.size();
    int num_patterns = get_num_patterns();
    word_t answer_word = words[answer];
    word_t opener_word = words[opener];

    // Initialize the prior vectors to be destructively modified by the solver
    std::vector<float> candidate_priors(num_words, 0.0f); // Statically sized
    std::vector<float> answer_priors(num_words, 0.0f);    // Dynamically sized
    std::vector<int> indecies(num_words, 0);

    // Perform the first round of update according to the opener word:
    // Performed by all threads as a means to initialize the candidate and 
    // answer priors
    coloring_t feedback = word_cmp(opener_word, answer_word);
    int words_remaining = 0;
    for(int j = 0; j < num_words; j++){ 
        if(word_cmp(opener_word, words[j]) == feedback){ // Underlying word is consistent with the guess
            prior_sum += priors[j];
            candidate_priors[j] = priors[j];
            answer_priors[words_remaining] = priors[j];
            indecies[words_remaining] = j; // Keep track of the original indecies
            words_remaining += 1;
        }
    }
    // The following should be unchanged from the functions above


    // Scratch work and entrypy storage. Private to each thread
    std::vector<float> probability_scratch;
    std::vector<float> entropys(work_size, 0.0f);

    
    // MPI Communication Buffers and data:
    int guess; // The index (Relative to the word list) of each thread's guess word
    float guess_score; // The score of each thread's guessed word
    std::vector<int> all_guesses(num_proc);
    std::vector<float> all_scores(num_proc); 


    MPI_Barrier(COMM);

    int iters = 1; // One round of game has already passed.

    while(iters < MAXITERS){
        /******************** Entropy Computation Phase **********************/
        if(words_remaining <= 2){ 
            // Random guess if there are no more than 2 valid words
            guess = arg_max(candidate_priors);
            guess_score = candidate_priors[guess];
        }
        else{ // More than 2 words: Compute the entropy for ALL words
            // Use thread local indexing
            for(int lcl_idx = 0; lcl_idx < work_size; lcl_idx++){
                // Zero out the probability scratch vector for new candidate
                probability_scratch.assign(num_patterns, 0.0f);

                // Computes the global word List index:
                int candidate_idx = lcl_idx + work_start;
                word_t candidate = words[candidate_idx];
                // Compute the coloring probability scratch vector on the fly:
                for(int j = 0; j < words_remaining; j++){
                    int answer_idx = indecies[j];
                    int weight = answer_priors[j];
                    coloring_t tmp_feedback = word_cmp(candidate, words[answer_idx]);
                    // Pool the prior weights into the probability scratch
                    probability_scratch[static_cast<int>(tmp_feedback)] += weight;
                }
                // Normalize the pooled weights, then compute the entropy score
                entropys[lcl_idx] = entropy_compute(probability_scratch, 
                    prior_sum) + (candidate_priors[candidate_idx] / prior_sum);
            }
            // Find the word that maximizes the expected entropy entropy.
            guess = arg_max(entropys);
            guess_score = entropys[guess];
            guess += work_start; // Convert local index to global index.
        }

        // Communicate Each thread's local result
        MPI_Allgather((void *) &guess, sizeof(int), MPI_BYTE, 
            (void*) &(all_guesses.front()), sizeof(int), MPI_BYTE, COMM);
        MPI_Allgather((void *) &guess_score, sizeof(float), MPI_BYTE, 
            (void*) &(all_scores.front()), sizeof(float), MPI_BYTE, COMM);
        // Select Word for all threads. selection_root is the thread

        guess = all_guesses[arg_max(all_scores)];


        /******************** Update Phase **********************/
        MPI_Barrier(COMM); // All threads must agree upon which thread has the guessed word.
        // Instead of serializing the update phase and communicate the result,
        // Each thread could just do the same set of redundant operations

        // Compute the feedback of the guessed word:
        word_t guess_word = words[guess];
        prior_sum = 0.0f;
        feedback = word_cmp(guess_word, answer_word);
        int write_idx = 0; // Used to push the remaining valid words to the front

        /**
         * For the candidate_priors: simply zero out the invalid words
         * For the answer_priors: push the prior weights of valid words
         *      to the front of the array
         * For the index vector: Do similar operations as the answer priors list
        */
        for(int j = 0; j < words_remaining; j++){ 
            // Only consider the words that are left:
            int remain_idx = indecies[j];
            coloring_t tmp_feedback = word_cmp(guess_word, words[remain_idx]);
            if(tmp_feedback == feedback){ // Underlying word is consistent with the guess
                prior_sum += answer_priors[j];
                answer_priors[write_idx] = answer_priors[j];
                indecies[write_idx] = indecies[j];
                write_idx += 1;
            }
            else{// Word is eliminated
                candidate_priors[indecies[j]] = 0.0f;
            }
        }
        words_remaining = write_idx; // Update the number of words remaining

        iters ++;
        // All threads should escape the loop if the guess is correct.
        if(is_correct_guess(feedback)) return iters;

        // Barrier at the end of the iteration.
        MPI_Barrier(COMM);
    }
    return iters;
}



int main(int argc, char **argv) {
    auto init_start = timestamp;
    int pid;
    int num_proc;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(COMM, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(COMM, &num_proc);



    // Initialization Stage
    std::string text_filename;
    std::string test_filename;
    std::string prior_filename;
    bool verbose = false;
    bool rand_prior = false;
    int opt;
    char mode = 'D';
    // Read program parameters
    while ((opt = getopt(argc, argv, "M:f:p:t:m:rv")) != -1) {
        switch (opt) {
        case 'f':
            text_filename = optarg;
            break;
        case 't':
            test_filename = optarg;
            break;
        case 'p':
            prior_filename = optarg;
            break;
        case 'm':
            wordlen = atoi(optarg);
            break;
        case 'r':
            rand_prior = true;
            break;
        case 'M':
            mode = *optarg;
            break;
        case 'v':
            verbose = true;
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }
    if(empty(text_filename)){
        usage(argv[0]);
        MPI_Finalize();
        exit(1);
    }
    if(wordlen <= 0 || wordlen > MAXLEN){
        std::cerr << "Invalid Wordlen Parameter [" << wordlen << "]\n";
        
    }

    // Initialize word list (Only Master)
    wordlist_t words;
    priors_t priors;
    std::vector<int> test_set;
    float priors_sum;
    args_t program_args;
    std::vector<std::vector<coloring_t>> pattern_matrix; // Will initialize later
    if(pid == 0){
        // Read Possible Word list
        int status = read_words_from_file(text_filename, words);
        if(status){
            MPI_Finalize();
            exit(1);
        }
        // Read prior list, if provided
        if(empty(prior_filename))
            if(rand_prior){
                priors = generate_random_priors(words.size(), priors_sum);
            } 
            else{
                priors = generate_uniform_priors(words.size(), priors_sum);
            }
        else{
            status = read_priors_from_file(prior_filename, priors_sum, priors);
            if(status){
                MPI_Finalize();
                exit(1);
            }
        }
        // Read test set, if provided
        if(!empty(test_filename)){ // Read test set from file
            status = read_test_set_from_file(test_filename, words, test_set);
            if(status){
                MPI_Finalize();
                exit(1);
            }
        }
        else{ // Test set is not provided, ask user for input.
            test_set.resize(1);
            std::string linebuf;
            word_t buffer;
            std::cout << "Test set is not provied. Please manually enter the answer word:\n";
            while(1){
                std::getline(std::cin, linebuf);
                if(linebuf.empty()) continue;
                str2word(linebuf, buffer);
                test_set[0] = list_query(words, buffer);
                if(test_set[0] < 0){
                    std::cout << "The word '";
                    word_print(buffer, 0, 0x20);
                    std::cout << "' is not valid.\n";
                }
                else break;
            }
        }
        // Master thread prepares for program argument broadcase
        program_args.wordlist_size = words.size();
        program_args.priors_sum = priors_sum;
        program_args.testset_size = test_set.size();
    }

    MPI_Bcast((void*) &program_args, sizeof(args_t), MPI_BYTE, 0, COMM);

    if(pid != 0){
        // Initialize non-master thread storage, prepare for broadcast
        words.resize(program_args.wordlist_size);
        priors.resize(program_args.wordlist_size);
        test_set.resize(program_args.testset_size);
        priors_sum = program_args.priors_sum;
    }
    // Obtain work region assignment on the coloring matrix
    int work_start;
    int work_end;
    task_split(static_cast<int>(words.size()), num_proc, pid, work_start, work_end);

    // Make sure memory space is properly allocated before broadcasting lists
    MPI_Barrier(COMM);

    // Broadcast the word list, prior weights and the test set
    MPI_Bcast((void*) &(words.front()), 
        sizeof(word) * program_args.wordlist_size, MPI_BYTE, 0, COMM);
    MPI_Bcast((void*) &(priors.front()),
        sizeof(float) * program_args.wordlist_size, MPI_BYTE, 0, COMM);
    MPI_Bcast((void *) &(test_set.front()), 
        sizeof(int) * program_args.testset_size, MPI_BYTE, 0, COMM);
    
    // Allocate space for the coloring matrix
    pattern_matrix.resize(work_end - work_start);
    for(auto &row : pattern_matrix){
        row.resize(program_args.wordlist_size);
    }

    // IO Complete
    auto init_end = timestamp;
    if(pid == 0)
        std::cout << "Initialization: " << TIME(init_start, init_end) << "\n";


    MPI_Barrier(COMM);

    int answer_index;
    int rounds;
    double rounds_total = 0.0;
    double time_total;
    if(mode == 'D'){ // Default MPI Mode
        auto precompute_start = timestamp;
        // Precompute the coloring matrix
        MPI_compute_patterns(pattern_matrix, words, work_start, work_end);
        auto precompute_end = timestamp;

        if(pid == 0)
            std::cout << "Pre-processing: " << TIME(precompute_start, precompute_end) << "\n";

        // Synchronize before stepping into the main solver loop.
        MPI_Barrier(COMM);


        // Benchmark all words in the test set.
        // Requires a deep copy for the priors in each benchmark
        priors_t prior_compute(priors.size());

        auto answer_start = timestamp;
        for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
            std::copy(priors.begin(), priors.end(), prior_compute.begin());
            answer_index = test_set[i];
            if(pid == 0){
                std::cout << "Benchmarking word: ";
                word_print(words[answer_index], 0, ' ');
            }
            if(verbose){
                rounds = solver_verbose(words, prior_compute, pattern_matrix, answer_index, priors_sum, work_start, work_end);
            }
            else{
                rounds = solver(prior_compute, pattern_matrix, answer_index, priors_sum, work_start, work_end);
            }
            MPI_Barrier(COMM); // A conservative synchronization after each benchmark
            if(pid == 0){
                std::cout << "<" << rounds << ">\n" << std::flush;
                rounds_total += static_cast<double>(rounds);
            }
        }

        auto answer_end = timestamp;
        time_total = TIME(answer_start, answer_end);
        if(pid == 0){
            double average_time = time_total / static_cast<double>(test_set.size());
            std::cout << "Average time taken: " << average_time << " sec per word\n";
            std::cout << "Average rounds per game: " << (rounds_total / static_cast<double>(test_set.size())) << "\n";
        }
    }
    else if(mode == 'O'){ // On the fly computation without pre-processing
        auto answer_start = timestamp;    
        for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
            answer_index = test_set[i];
            if(pid == 0){
                std::cout << "(OTF) Benchmarking word: ";
                word_print(words[answer_index], 0, ' ');
            }
            if(verbose)
                rounds = solver_verbose_no_precompute(words, priors, answer_index, priors_sum);
            else
                rounds = solver_no_precompute(words, priors, answer_index, priors_sum);
            MPI_Barrier(COMM); // A conservative synchronization after each benchmark
            if(pid == 0){
                std::cout << "<" << rounds << ">\n" << std::flush;
                rounds_total += static_cast<double>(rounds);
            }
        }


        auto answer_end = timestamp;
        time_total = TIME(answer_start, answer_end);

        if(pid == 0){
            double average_time = time_total / static_cast<double>(test_set.size());
            std::cout << "Average time taken: " << average_time << " sec per word\n";
            std::cout << "Average rounds per game: " << (rounds_total / static_cast<double>(test_set.size())) << "\n";
        }

    }
    else if(mode == 'P'){
        auto precompute_start = timestamp;
        int opener = opener_precompute(words, priors, priors_sum);
        auto precompute_end = timestamp;

        if(pid == 0){
            std::cout << "Opener Selection: " << TIME(precompute_start, precompute_end) << "\n";
            std::cout << "Opener Word: ";
            word_print(words[opener], get_num_patterns() - 1);
        }
        auto answer_start = timestamp;    

        // Test Solver no Precompute
        for (int i = 0; i < static_cast<int>(test_set.size()); i ++){
            answer_index = test_set[i];
            if(pid == 0){
                std::cout << "(OTF) Benchmarking word: ";
                word_print(words[answer_index], 0, ' ');
            }
            if(verbose)
                rounds = solver_OTF_opener_verbose(words, priors, answer_index, opener);
            else
                rounds = solver_OTF_opener(words, priors, answer_index, opener);
            MPI_Barrier(COMM); // A conservative synchronization after each benchmark
            if(pid == 0){
                std::cout << "<" << rounds << ">\n" << std::flush;
                rounds_total += static_cast<double>(rounds);
            }
        }
        auto answer_end = timestamp;
        time_total = TIME(answer_start, answer_end);

        if(pid == 0){
            double average_time = time_total / static_cast<double>(test_set.size());
            std::cout << "Average time taken: " << average_time << " sec per word\n";
            std::cout << "Average rounds per game: " << (rounds_total / static_cast<double>(test_set.size())) << "\n";
        }
    }
    else{
        if(pid == 0){
            std::cout << "Invalid Mode, aborting program" << "\n" << std::flush;
        }
    }

    
    


    MPI_Finalize();
    return 0;
}
