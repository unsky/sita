//---------------------------------
//write by unsky
//---------------------------------
#ifndef SITA_STUFF_WORKSPACE_H_
#define SITA_STUFF_WORKSPACE_H_
#include <vector>
#include <map>
#include <set>
#include <glog/logging.h>
#include <cuda_runtime.h>
#include "tensor.h"
#include "context.h"
#include "macros.h"
#include "graph.h"
#include "registry.h"
#include "operator.h"
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
    GlobalWorkSpace():WorkSpace(){};
    ~GlobalWorkSpace(){};

    //temp tensor
    std::pair<int, Tensor<Dtype> * > fetch_temp_tensor();

    void release_temp_tensor(int released_id);

    float temp_tensor_memory_size();

    //flow tensor
    void init_input(std::string name);

    void init_output(std::string name);

    Tensor<Dtype>* fetch_input(std::string name);

    Tensor<Dtype>* fetch_output(std::string name);

    std::string flow_tensor_list();
    

    //params
    void init_param(std::string op_name, std::string op_type, std::string param_name, std::vector<int> shape, Filler);

    Tensor<Dtype>* fetch_param(std::string op_name, std::string param_name);

    std::string param_list();



    //grap
    inline void build_graph(Graph * graph){
        _graph = graph;
        graph_show();
    };
    inline void graph_show(){
        _graph->graph_symbol_show();
    }

    void global_init();
    void forward();
    void backward();
    void train();

private:
    // temp_tensor bool true: using  false:released
    std::vector<std::pair<Tensor<Dtype> *, bool> > _temp_tensor_control;
    std::vector<Tensor<Dtype> > _temp_tensor;
    //graph
    Graph * _graph;
    std::vector<boost::shared_ptr<Operator<Dtype> > > _ops;
    // input/output name
    std::map<std::string, Tensor<Dtype> > _flow_tensor;
    // params         name                   type          weight/bias name  weight/bias
    std::map<std::string, OperatorParam<Dtype> > _params;
    DISABLE_COPY_AND_ASSIGN(GlobalWorkSpace);
};



}//namespce
#endif //SITA_STUFF_WORKSPACE
