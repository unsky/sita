//---------------------------------
//write by unsky
//---------------------------------
#ifndef SITA_STUFF_TENSOR_H_
#define SITA_STUFF_TENSOR_H_

#include <vector>
#include <iostream>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include <climits>

#include "sita/stuff/memory_control.h"
#include "sita/stuff/macros.h"
#include "sita/stuff/context.h"

namespace  sita {

template <typename Dtype>
class Tensor {
public:

    Tensor(): _count(0), _dim(0), _data(), _diff() {
        _shape.clear();
        _shape.push_back(0);
    }

    explicit Tensor(const int num, const int channels, const int height,
                    const int width);

    explicit Tensor(const std::vector<int >& shape);
    // fetch data ptr from gpu or cpu
    inline Dtype* mutable_cpu_data() {
        return (Dtype*)_data->mutable_cpu_data();
    }
    inline Dtype* mutable_gpu_data() {
        return (Dtype*)_data->mutable_gpu_data();
    }
    inline const  Dtype* cpu_data() {
        return (const Dtype*)_data->cpu_data();
    }
    inline const Dtype* gpu_data() {
        return (const Dtype*)_data->gpu_data();
    }

    //fetch diff ptr from gpu or cpu
    inline Dtype* mutable_cpu_diff() {
        return (Dtype*)_diff->mutable_cpu_data();
    }
    inline Dtype* mutable_gpu_diff() {
        return (Dtype*)_diff->mutable_gpu_data();
    }
    inline const  Dtype* cpu_diff() {
        return (const Dtype*)_diff->cpu_data();
    }
    inline const Dtype* gpu_diff() {
        return (const Dtype*)_diff->gpu_data();
    }

    inline const std::vector<int >shape() const{
        return _shape;
    }

    inline std::string shape_string() const {
        std::ostringstream stream;

        for (int i = 0; i < _shape.size(); ++i) {
            stream << _shape[i] << " ";
        }

        stream << "(" << _count << ")";
        return stream.str();
    }

    inline const int dim() const {
        return _dim;
    }

    inline const int count() const {
        return _count;
    }

    void reshape(const std::vector<int> &shape);
    void reshape(const int num, const int channels, const int height, const int width);
    void reshape_like(const Tensor<Dtype> &t_other);

    void copy_from(const Tensor<Dtype> &t_other, bool reshape = true);

    void set_data_zero();
    void set_diff_zero();

   int get_site_by_coord(const int num, const int channels, const int height, const int width);
   int get_site_by_coord(const std::vector<int > &coord);

protected:
    std::vector<int > _shape;
    int _count;
    int _dim;
    boost::shared_ptr<MemControl> _data;
    boost::shared_ptr<MemControl> _diff;

  //  DISABLE_COPY_AND_ASSIGN(Tensor);
};



}//namespace sita


#endif //SITA_STUFF_TENSOR_H_
