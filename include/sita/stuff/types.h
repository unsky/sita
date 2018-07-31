//
// Created by cs on 25/07/18.
//

#ifndef SITA_STUFF_TYPES_HPP
#define SITA_STUFF_TYPES_HPP
#include <vector>
#include <string>
#include "tensor.h"
namespace  sita{

struct OperatorConfig{
    bool is_loss_op;
    bool is_data_op;
    bool has_param_op;
};

struct Filler{
    std::string type;

};

template <typename Dtype>
struct OperatorParam{
    std::string type;
    std::vector<Tensor<Dtype> > params;
    std::vector<Filler> filler;
};

};

#endif //SITA_TYPES_H
