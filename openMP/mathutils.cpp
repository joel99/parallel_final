#include "mathutils.h"
#include <omp.h>

bool is_zero(float x){
    return std::fabs(x) <= PRECISION;
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