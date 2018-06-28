#include "sita/stuff/tensor.h"
#include <cuda_runtime.h>
namespace  sita{
    template <typename Dtype>
    Tensor<Dtype>::Tensor(const std::vector<int > &shape){
        _shape.clear();
        int _count = 0;
        int count = 1;
        for(int i = 0; i < shape.size(); i++){
            _shape.push_back(shape[i]);
            count *= shape[i];
        }

        _dim = shape.size();
        _count = count;
        _data.reset(new MemControl(_count * sizeof(Dtype)));
        _diff.reset(new MemControl(_count * sizeof(Dtype)));

    }
    template <typename Dtype>
    void Tensor<Dtype>::reshape(std::vector<int> shape){

        //CUDA_CHECK(cudaMalloc(&_ptr_gpu, _size));

    }


}//namespace