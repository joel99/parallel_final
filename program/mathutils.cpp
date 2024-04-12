#include "mathutils.h"


float entropy(float prob){
    if(fabs(prob) <= PRECISION) return 0.0;
    return prob * log2f(1.0f/prob);
}

void scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out, float multiplier){
    size_t n = index.size();
    index_t j;
    for(size_t i = 0; i < n; i++){
        j = index[i];
        out[j] += multiplier * in[i];
    }
}

void masked_scatter_reduce(std::vector<index_t> &index, std::vector<float> &in,
    std::vector<float> &out, std::vector<bool> &mask, float multiplier){
    size_t n = index.size();
    index_t j;
    for(size_t i = 0; i < n; i++){
        j = index[i];
        out[j] += multiplier * in[i] * mask[i];
    }
}

float map_reduce_sum(std::vector<float> vec, float (*f) (float)){
    float out = 0.0;
    for(size_t i = 0; i < vec.size(); i++){
        out += f(vec[i]);
    }
    return out;
}
