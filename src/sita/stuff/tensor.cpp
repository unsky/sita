//
// Created by unsky on 29/06/18.
//
#include "sita/stuff/tensor.h"

namespace sita {

template<typename Dtype>
Tensor<Dtype>::Tensor(const int num, const int channels, const int height,
                      const int width) {
    _shape.clear();
    _shape.push_back(num);
    _shape.push_back(channels);
    _shape.push_back(height);
    _shape.push_back(width);
    _count = 1;
    for(int i = 0; i < _shape.size(); i++){
        CHECK_LT(_count, INT_MAX/_shape[i]) << "num of tensor element should less than INT_MAX";
        _count *= _shape[i];

    }
    _dim = 4;
    _data.reset(new MemControl(_count * sizeof(Dtype)));
    _diff.reset(new MemControl(_count * sizeof(Dtype)));
}

template<typename Dtype>
Tensor<Dtype>::Tensor(const std::vector<int >& shape) {
    _shape.clear();
    CHECK_GE(shape.size(), 1) << "dim of tensor should greater equal 1";
    for(int i = 0; i <shape.size(); i++){
        CHECK_GT(shape[i], 0) << "axis of tensor should greater than 0";
    }

    _count = 0;
    int count = 1;

    for (int i = 0; i < shape.size(); i++) {
        _shape.push_back(shape[i]);
        CHECK_LT(count, INT_MAX/shape[i]) << "num of tensor element should less than INT_MAX";
        count *= shape[i];
    }

    _dim = shape.size();
    _count = count;
    _data.reset(new MemControl(_count * sizeof(Dtype)));
    _diff.reset(new MemControl(_count * sizeof(Dtype)));
}

template<typename Dtype>
void Tensor<Dtype>::reshape(const std::vector<int> &shape) {
    CHECK_GE(shape.size(), 1) << "dim of tensor should greater or equal to 1";
    for(int i = 0; i <shape.size(); i++){
        CHECK_GT(shape[i], 0) << "axis of tensor should greater to 0";
    }

    int count = 1;
    for(int i = 0; i < shape.size(); i++){
        CHECK_LT(count, INT_MAX/shape[i]) << "num of tensor element should less than INT_MAX";
        count *= shape[i];
    }

    if(count > _count){
        _data.reset(new MemControl(count * sizeof(Dtype)));
        _diff.reset(new MemControl(count * sizeof(Dtype)));
    }

    _count = count;
    _dim = shape.size();

    _shape.clear();
    for(int i = 0; i < shape.size(); i++){
        _shape.push_back(shape[i]);
    }


}

template<typename Dtype>
void Tensor<Dtype>::reshape(const int num,const int channels,const int height,const int width){

    std::vector<int> shape;
    shape.push_back(num);
    shape.push_back(channels);
    shape.push_back(height);
    shape.push_back(width);

    reshape(shape);
}

template<typename Dtype>
void Tensor<Dtype>::reshape_like(const Tensor<Dtype> &t_other){
    reshape(t_other.shape());
}

template<typename Dtype>
void Tensor<Dtype>::copy_from(const Tensor<Dtype> &t_other, bool reshape = true){
    if(reshape) {
        reshape_like(t_other);
    }else{
        CHECK_EQ(_dim, t_other.dim()) << "reshape is false, but the dim of dst and src tensor is not equal";
        for(int i = 0; i < _shape.size(); i++) {
            CHECK_EQ(_shape[i], t_other.shape()[i])<< "the shape of dst and src is not equal";
        }
    }
    //data diff cpu and gpu
    memcpy(this->mutable_cpu_data(), t_other.cpu_data(), sizeof(Dtype) * _count);
    memcpy(this->mutable_cpu_diff(), t_other.cpu_diff(), sizeof(Dtype) * _count);

    CUDA_CHECK(cudaMemcpy(this->mutable_gpu_data(), t_other.gpu_data(), sizeof(Dtype) * _count, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(this->mutable_gpu_diff(), t_other.gpu_diff(), sizeof(Dtype) * _count, cudaMemcpyDeviceToDevice));

}
template<typename Dtype>
void Tensor<Dtype>::set_data_zero(){
    CHECK_GT(_count, 0) << "tensor count must greater than 0 when set to 0";

    //data cpu and gpu
    memset(this->mutable_cpu_data(), 0, sizeof(Dtype) * _count);
    this->gpu_data();
}

template<typename Dtype>
void Tensor<Dtype>::set_diff_zero(){
    CHECK_GT(_count, 0) << "tensor count must greater than 0 when set to 0";

    //data cpu and gpu
    memset(this->mutable_cpu_diff(), 0, sizeof(Dtype) * _count);
    this->gpu_diff();
}

template<typename Dtype>
int Tensor<Dtype>::get_site_by_coord(const std::vector<int > &coord){
    CHECK_EQ(coord.size(), _dim) << "dim of coordinate is not equal to this tensor";
    int site = 0;
    for(int i = 0; i < coord.size() - 1; i++){
        CHECK_LE(coord[i], _shape[i] - 1) << "coordinate shold less than this tensor shape";
        int count = 1;
        for(int j = i + 1; j < _dim; j++ ){
            count *= _shape[j];
        }
        site = site + coord[i] * count;
    }
    site += coord[coord.size() - 1];
    return site;
}

template<typename Dtype>
int Tensor<Dtype>::get_site_by_coord(const int num, const int channels, const int height, const int width){
    std::vector<int > coord;
    coord.push_back(num);
    coord.push_back(channels);
    coord.push_back(height);
    coord.push_back(width);
    return get_site_by_coord(coord);
}
template<typename Dtype>
void Tensor<Dtype>::clear()
{
    _count = 0;
    _dim = 0;
    _data.reset();
    _diff.reset();
}

INSTANTIATE_CLASS(Tensor);
}