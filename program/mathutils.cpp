#include "mathutils.h"

bool is_zero(float x){
    return std::fabs(x) <= PRECISION;
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

float entropy_reduce(std::vector<float> floats, float normalize){
    float out = 0.0;
    for(size_t i = 0; i < floats.size(); i++){
        out += normalize_entropy(floats[i], normalize);
    }
    return out;
}
