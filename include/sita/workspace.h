//---------------------------------
//write by unsky
//---------------------------------
#ifndef SITA_WORKSPACE_H
#define SITA_WORKSPACE_H
#include <vector>
#include <map>
#include <set>
#include <glog/logging.h>
#include <cuda_runtime.h>
#include "tensor.h"
#include "context.h"
#include "macros.h"
#include "sita/graph.h"
#include "types.h"
#include "sita/dataprovider/mnist_dataprovider.h"
#include "sita/dataprovider/dataprovider.h"
#include "sita/registry.h"
#include "sita/operator.h"
namespace sita{


class WorkSpace{
public:
    WorkSpace():_gpu_id(0){}

    ~WorkSpace(){}

    void device_query();

    void set_device(int gpu_id);
    inline int get_working_device(){
        return _gpu_id;
    }

protected:
    int _gpu_id;
};

template <typename Dtype>
class Operator;


template <typename Dtype>
class GlobalWorkSpace : public WorkSpace{

public:
    GlobalWorkSpace():WorkSpace(){
        _temp_tensor_control.resize(PRE_TEMP_TENSOR_NUM);
        _temp_tensor.resize(PRE_TEMP_TENSOR_NUM);
        _temp_tensor_num = 0;
    };
    ~GlobalWorkSpace(){};

    //temp tensor
    TempTensor<Dtype> new_tensor();
    void free_tensor(TempTensor<Dtype> t);
    void temp_memory();

    //flow tensor
    void init_input(std::string name);
    void init_output(std::string name);
    Tensor<Dtype>* fetch_input(std::string name);
    Tensor<Dtype>* fetch_output(std::string name);
    std::string flow_tensor_list();
    void flow_memory();

    //params
    void init_param(std::string op_name, std::string op_type, std::string param_name, std::vector<int> shape,
                    ParamConfig p_config, bool is_shared);
    Tensor<Dtype>* fetch_param(std::string op_name, std::string param_name, bool is_shared);
    std::string param_list();
    void param_memory();


    //graph
    inline void graph_show(){
        _graph->graph_symbol_show();
    }

    void global_init(Graph * graph, DataProvider<Dtype> * data_provider);
    void forward();
    void backward();
    void train();

private:
    // temp_tensor bool true: using  false:released
    std::vector<std::pair<Tensor<Dtype> *, bool> > _temp_tensor_control;
    std::vector<Tensor<Dtype> > _temp_tensor;
    int _temp_tensor_num;
    //graph
    Graph * _graph;
    std::vector<boost::shared_ptr<Operator<Dtype> > > _ops;
    // input/output name
    std::map<std::string, Tensor<Dtype> > _flow_tensor;
    // params      name        type       weight/bias name  weight/bias
    std::map<std::string, OperatorParam<Dtype> > _params;

    DataProvider<Dtype> * _data_provider;
    DISABLE_COPY_AND_ASSIGN(GlobalWorkSpace);
};



}//namespce
#endif //SITA_WORKSPACE
