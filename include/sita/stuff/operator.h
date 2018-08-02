//
// Created by unsky on 03/07/18.
//

#ifndef SITA_STUFF_OPERATORS_H
#define SITA_STUFF_OPERATORS_H
#include <vector>
#include <string>
#include "macros.h"
#include "workspace.h"
#include "sita_parameter.h"
#include "types.h"
namespace sita{



template <typename Dtype>
class GlobalWorkSpace;

template <typename Dtype>
class Operator{
public:
    Operator(const OperatorDef& opdef, GlobalWorkSpace<Dtype> *gws):_opdef(opdef),_gws(gws){}
    ~Operator(){}
    void setup();
    void init_param(std::string param_name, std::vector<int> shape);

    Tensor<Dtype> * fetch_input(std::string name);
    Tensor<Dtype> * fetch_output(std::string name);
    Tensor<Dtype> * fetch_param(std::string name);

    virtual void  init(){};
    virtual void forward(){};
    virtual void backward(){};
protected:
    GlobalWorkSpace<Dtype> *_gws;
    OperatorDef _opdef;
    Filler _filler;
    std::vector<std::string> _inputs;
    std::vector<std::string> _outputs;
    std::vector<std::string> _params;
};

}//namespace


#endif //SITA_STUFF_OPERATORS_H
