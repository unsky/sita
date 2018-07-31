//
// Created by cs on 31/07/18.
//

#ifndef SITA_STUFF_SITA_PARAMETER_H
#define SITA_STUFF_SITA_PARAMETER_H

#include "types.h"
namespace  sita {

struct AddOpParameter {
    int stride_h;
    int stride_w;
};

struct SitaParameter {
    AddOpParameter add_op_param;
    Filler filler;
};
struct OperatorDef{
    std::string name;
    std::string type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    SitaParameter param;
};
struct GraphSym{
    std::string graph_name;
    std::vector<OperatorDef > ops;
};

}//namespace

#endif //SITA_OPERATORPARAMTER_H
