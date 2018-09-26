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
    Operator(const OperatorParameter& opdef, GlobalWorkSpace<Dtype> *gws, std::string phase):_opdef(opdef),_gws(gws),_phase(phase){}
    ~Operator(){}
    void setup();
    void init_param(std::string param_name, std::vector<int> shape, ParamConfig p_config);
    Tensor<Dtype> * fetch_input(std::string name);
    Tensor<Dtype> * fetch_output(std::string name);
    Tensor<Dtype> * fetch_param(std::string name);
    inline std::string operator_name(){
        return _opdef.name();
    }
    virtual void  init(){};
    virtual void  infer_shape(){};
    virtual void forward(){};
    virtual void backward(){};
protected:
    GlobalWorkSpace<Dtype> *_gws;
    OperatorParameter _opdef;
    std::vector<ParamConfig> _param_configs;
    std::vector<std::string> _inputs;
    std::vector<std::string> _outputs;
    std::vector<std::string> _params;
    std::map<std::string, std::vector<int> > _input_shapes;
    std::map<std::string, std::vector<int> > _output_shapes;
    std::map<std::string, std::vector<int> > _param_shapes;
    bool _is_shared;
    std::vector<std::pair<std::string, std::string> > _shared_param_pairs;
    bool _gradient_block;
    std::string _phase;
    std::string _op_name;
};

}//namespace


#endif //SITA_STUFF_OPERATORS_H
