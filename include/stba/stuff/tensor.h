//---------------------------------
//write by unsky
//---------------------------------
#ifndef STBA_STUFF_TENSOR_H_
#define STBA_STUFF_TENSOR_H_

#include <vector>
#include <isotream>
#include <sstream>
#include <boost/shared_ptr.hpp>

#include "stba/stuff/memcontrol.h"
#include "stba/stuff/common.h"

namespace  stba{
    template <typedef Dtype>
    class Tensor {
        public:

            Tensor(){};

            Tensor(int num, int channel, int height, int width);

            Tensor(std::vector<int > shape);

            inline string shape_string() const {
                ostringstream stream;
                for (int i = 0; i < _shape.size(); ++i) {
                    stream << _shape[i] << " ";
                }
                stream << "(" << _count << ")";
                return stream.str();
            }

            // fetch data ptr from gpu or cpu
            inline Dtype * mutable_cpu_data(){
                return (Dtype*)_data->mutable_cpu_data();
            }
            inline Dtype *mutable_gpu_data(){
                return (Dtype*)_data->mutable_gpu_data();
            }
            inline const  Dtype *cpu_data(){
                return (const Dtype *)_data->cpu_data();
            }
            inline const Dtype *gpu_data(){
                return (const Dtype *)_data->gpu_data();
            }

            //fetch diff ptr from gpu or cpu
            inline Dtype * mutable_cpu_diff(){
                return (Dtype*)_diff->mutable_cpu_data();
            }
            inline Dtype *mutable_gpu_diff(){
                return (Dtype*)_diff->mutable_gpu_data();
            }
            inline const  Dtype *cpu_diff(){
                return (const Dtype *)_diff->cpu_data();
            }
            inline const Dtype *gpu_diff(){
                return (const Dtype *)_diff->gpu_data();
            }

        protected:
            std::vector<int > _shape;
            int _count;
            int _dim;
            shared_ptr<MemControl> _data;
            shared_ptr<MemControl> _diff;
            Dtype * _data;
            DISABLE_COPY_AND_ASSIGN(Tensor);
    };



}//namespace stba


#endif //STBA_STUFF_TENSOR_H_
