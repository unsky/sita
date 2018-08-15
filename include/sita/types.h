//
// Created by unsky on 25/07/18.
//

#ifndef SITA_TYPES_HPP
#define SITA_TYPES_HPP
#include <vector>
#include <string>
#include <map>
#include "tensor.h"
#include "sita/proto/sita.h"
namespace  sita{


template <typename Dtype>
struct OperatorParam{
    std::string type;
    std::map<std::string, Tensor<Dtype> > params;
    std::map<std::string, FillerParameter> fillers;
};

};

#endif //SITA_TYPES_H
