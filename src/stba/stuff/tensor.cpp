#include "stba/stuff/tensor.h"
namespace  stba{
    template <typename Dtype>
    Tensor<Dtype>::Tensor(const int num, const int channels, const int height,
                              const int width) {
        _shape.clear();
        _shape.push_back(num);
        _shape.push_back(channels);
        _shape.push_back(height);
        _shape.push_back(width);
        _count = num * channels * height * width;
        _dim = 4;
    }
    template  <typename Dtype>
    Tensor<Dtype>::Tensor(vector<int > shape){
        _shape.clear();
        int _count = 0;
        int count = 1;
        for(int i = 0; i < shape.size(); i++){
            _shape.push_back(shape[i]);
            count *= shape[i];
        }
        _dim = shape.size();
        _count = count;
    }





}//namespace