//
// Created by cs on 31/07/18.
//

#ifndef SITA_STUFF_SITA_PARAMETER_H
#define SITA_STUFF_SITA_PARAMETER_H

#include "types.h"
namespace  sita {

struct AddOpParameter {
    int stride_h = 1;
    int stride_w =1;
};
struct DataTestParameter{
    int batch_size = 1;
};

struct SitaParameter {
    AddOpParameter add_op_param;
    DataTestParameter data_test_param;
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
