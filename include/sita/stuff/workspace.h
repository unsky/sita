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
namespace sita{


class WorkSpace{
public:
    WorkSpace(){}

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
class GlobalWorkSpace : public WorkSpace{

public:
    GlobalWorkSpace():WorkSpace(){};
    ~GlobalWorkSpace(){};

    //temp tensor

    std::pair<int, Tensor<Dtype> * > fetch_temp_tensor();
    void release_temp_tensor(int released_id);
    float temp_tensor_memory_size();

    //grap
    void build_grap(){};
    void train(){};

    //net

private:
    // temp_tensor bool true: using  false:released
    std::vector<std::pair<Tensor<Dtype> *, bool> > _temp_tensor_control;
    std::vector<Tensor<Dtype> > _temp_tensor;



    DISABLE_COPY_AND_ASSIGN(GlobalWorkSpace);
};



}//namespce
#endif //SITA_STUFF_WORKSPACE