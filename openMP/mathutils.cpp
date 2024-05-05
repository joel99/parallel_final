#include "mathutils.h"
#include <omp.h>
#include <list>

bool is_zero(float x){
    return std::fabs(x) <= PRECISION;
}

unsigned long ceil_xdivy(unsigned long X, unsigned long Y){
    return (X + (Y - 1)) / Y;
}

// https://stackoverflow.com/questions/686353/random-float-number-generation
float f_rand(float low, float high){
    return static_cast <float> (rand()) /
        ( static_cast <float> (RAND_MAX/(high-low)));
}

float entropy(float prob){
    if(is_zero(prob)) return 0.0;
    return prob * log2f(1.0f/prob);
}

float normalize_entropy(float in, float total){
    return(entropy(in/total));
}

void scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out){
    size_t n = index.size();
    index_t j;
    for(size_t i = 0; i < n; i++){
        j = index[i];
        out[j] += in[i];
    }
}

void masked_scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out, std::vector<bool> &mask, float multiplier){
    size_t n = index.size();
    index_t j;
    for(size_t i = 0; i < n; i++){
        j = index[i];
        out[j] += in[i] * mask[i];
    }
}

/**
 * A reduction based scatter reduce. To be invoked in a parallel region.
 * @param scratch - a  <num_proc> * <data_out.size()> temporary matrix for thread
 *                  local aggregation (better than local allocation)
*/
void parallel_scatter_reduce(std::vector<index_t> &data_index,
                              std::vector<float> &data_in,
                              std::vector<float> &data_out){
                            //   std::vector<float> &data_out,
                            //   std::vector<std::vector<float>> &scratch){
    int n = static_cast<int>(data_in.size());
    int m = static_cast<int>(data_out.size());
    // Manual - overhead of scratch is too high
    // #pragma omp parallel // Local Aggregation Step
    // {
    //     int thread_id = omp_get_thread_num();
    //     int idx;
    //     #pragma omp for // Dynamic overhead is terrible here
    //     // #pragma omp for schedule(dynamic, 32)
    //         for(int i = 0; i < n; i++){
    //             idx = data_index[i];
    //             scratch[thread_id][idx] += data_in[i];
    //         }
    //         #pragma omp critical
    //         {
    //             for(int i = 0; i < m; i++){
    //                 data_out[i] += scratch[thread_id][i];
    //             }
    //         }
    // }

    // Directly as OMP pragma
    // #pragma omp parallel
    // {
    float* data_out_ptr = data_out.data();
    int idx;
    #pragma omp for reduction(+:data_out_ptr[:m])
    for(int i = 0; i < n; i++){
        idx = data_index[i];
        data_out_ptr[idx] += data_in[i];
    }
    // }
}

void scatter_reduce_cap(std::vector<float> &data_in, // input
                            std::vector<std::vector<short unsigned int>> &data_index, // output by input
                            std::vector<std::vector<float>> &data_out, // input by color/output 
                            std::vector<std::vector<std::vector<float>>> &scratch, // capacity x color/output
                            std::vector<omp_lock_t> &locks){ // thread by input by color/output
    /*
        Hybrid guess-candidate parallel, to demonstrate a point about candidate
        Here threads will track and be adding and attempting to reduce work queues accumulated across guesses.

        This version uses limited capacity scratch, so we cannot have arbitrarily large queues.
    */
    int guesses = static_cast<int>(data_index.size());
    int candidates = static_cast<int>(data_index[0].size());
    int colors = static_cast<int>(data_out[0].size());
    int capacity = static_cast<int>(scratch[0].size());
    // std::cout << "Guesses: " << guesses << " Colors: " << colors << " Capacity: " << capacity << "\n";

    int num_threads = omp_get_max_threads();
    // TODO, there's no reason only own thread should commit own work other than cache effect
    // TODO profile write/read ratio
    // thread by capacity to process, true if there's data in scratch to write
    std::vector<std::vector<bool>> task_mask = std::vector<std::vector<bool>>(num_threads, std::vector<bool>(capacity, false));
    // thread by work-item of (lane, guess) pairs to know what to map from/to
    auto task_queue = std::vector<std::list<std::pair<int, int>>>(
        num_threads, std::list<std::pair<int, int>>());

    // Hypothetical gains over guess-parallel if shared cache can be leveraged
    // Manual
    int candidate_span = ceil_xdivy(candidates, num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int read_min = candidate_span * thread_id;
        int read_max = std::min(read_min + candidate_span, candidates);

        int idx;

        for (int guess = 0; guess < guesses; guess++){
            // find the next empty slot
            int write_lane = -1;
            for (int i = 0; i < capacity; i++){
                if(!task_mask[thread_id][i]){
                    write_lane = i; 
                    scratch[thread_id][write_lane].assign(colors, 0.0); // clear
                    task_mask[thread_id][write_lane] = true;
                    break;
                }
            }
            // std::cout << "Thread: " << thread_id << " Writing Lane: " << write_lane << "\n";
            for(int candidate = read_min; candidate < read_max; candidate++){
                idx = data_index[guess][candidate];
                scratch[thread_id][write_lane][idx] += data_in[candidate];
            }
            task_queue[thread_id].push_back(std::make_pair(guess, write_lane));
            bool try_once = true;
            // std::cout << "Thread: " << thread_id << " Queue Size: " << task_queue[thread_id].size() << "\n";
            // attempt to clear accumulated work, iterate through list, and do not exceed capacity
            while (try_once || task_queue[thread_id].size() >= capacity) {
                auto it = task_queue[thread_id].begin();
                while (it != task_queue[thread_id].end()) {
                    int write_guess = it->first;
                    int write_lane = it->second;
                    // std::cout << "Thread: " << thread_id << " Guess / Lane " << write_guess << " / " << write_lane << "\n";
                    if (omp_test_lock(&locks[write_guess])) {
                        for (int color = 0; color < colors; color++) {
                            // std::cout << "Thread: " << thread_id << " color / lane " << color << " / " << write_lane << "\n";
                            data_out[write_guess][color] += scratch[thread_id][write_lane][color];
                        }
                        omp_unset_lock(&locks[write_guess]);
                        // std::cout << "Thread: " << thread_id << " Wrote Guess - free now: " << write_guess << "\n";
                        task_mask[thread_id][write_lane] = false;
                        it = task_queue[thread_id].erase(it); // Erase returns the next iterator
                    } else {
                        ++it;
                        // Only move to next element if task wasn't done (i.e., lock not acquired)
                    }
                }
                try_once = false;
            }
        }
        // clear queues - TODO assess how much is used here
        auto it = task_queue[thread_id].begin();
        while (it != task_queue[thread_id].end()) {
            auto pair = task_queue[thread_id].front();
            int write_guess = pair.first;
            int write_lane = pair.second;
            std::cout << "Thread: " << thread_id << " Writing Guess: " << write_guess << "\n";
            omp_set_lock(&locks[write_guess]);
            for (int color = 0; color < colors; color++) {
                data_out[write_guess][color] += scratch[thread_id][write_lane][color];
            }
            omp_unset_lock(&locks[write_guess]);
            ++it;
        }
    }
}


// Egregiously bad, 10x slower
// void parallel_scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
//     std::vector<float> &out){
//     size_t n = index.size();
//     index_t j;
//     #pragma omp parallel for private(j)
//     for(size_t i = 0; i < n; i++){
//         j = index[i];
//         #pragma omp atomic
//         out[j] += in[i];
//     }
// }

float entropy_compute(std::vector<float> floats, float normalize){
    float out = 0.0;
    for(size_t i = 0; i < floats.size(); i++){
        out += normalize_entropy(floats[i], normalize);
    }
    return out;
}

void parallel_entropy_compute(std::vector<float> floats, float normalize, float& out){
    #pragma omp single
    out = 0.0;
    #pragma omp for reduction(+:out)
    for(size_t i = 0; i < floats.size(); i++){ 
        out += normalize_entropy(floats[i], normalize);
    }
}