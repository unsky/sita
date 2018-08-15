//
// Created by unsky on 03/07/18.
//

#ifndef SITA_DLFLOW_OPERATORS_H
#define SITA_DLFLOW_OPERATORS_H
#include <vector>
#include <string>
#include "sita/macros.h"
#include "sita/workspace.h"
#include "sita/types.h"
#include "sita/proto/sita.h"
namespace sita{

template <typename Dtype>
class GlobalWorkSpace;

template <typename Dtype>
class Operator{
public:
    Operator(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws):_opdef(opdef),_gws(gws){}
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
    OperatorParameter _opdef;
    FillerParameter _filler;
    std::vector<std::string> _inputs;
    std::vector<std::string> _outputs;
    std::vector<std::string> _params;
};

}//namespace


#endif //SITA_STUFF_OPERATORS_H
