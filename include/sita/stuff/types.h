//
// Created by cs on 25/07/18.
//

#ifndef SITA_STUFF_TYPES_HPP
#define SITA_STUFF_TYPES_HPP
#include <vector>
#include <string>
#include <map>
#include "tensor.h"
namespace  sita{
struct Filler{
    std::string type;
};

template <typename Dtype>
struct OperatorParam{
    std::string type;
    std::map<std::string, Tensor<Dtype> > params;
    std::map<std::string, Filler> fillers;
};

};

#endif //SITA_TYPES_H
