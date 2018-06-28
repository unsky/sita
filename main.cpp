//
// Created by unsky on 28/06/18.
//
#include "sita/stuff/common.h"
#include "sita/stuff/memcontrol.h"
#include "sita/stuff/tensor.h"
#include <glog/logging.h>


int test_tensor()
{
    sita::Tensor<float> t(1,1,1,4);

}
int main(int argc, char** argv) {
    std::vector<int > shape;

    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(1);
    shape.push_back(3);

    boost::shared_ptr <sita::Tensor<float>> test1(new sita::Tensor<float>(shape));

    LOG(INFO) << test1->shape_string();

    const float *data1 = test1->cpu_data();

    for (int i = 0; i < test1->count(); i++) {
        LOG(INFO) << data1[i];
    }



     sita::Tensor<float> test2(1, 1, 1, 3);
     LOG(INFO) << test2.shape_string() << test2.count();
     float *data2 = test2.mutable_cpu_data();

     for (int i = 0; i < test2.count(); i++) {
         LOG(INFO) << data2[i];
          data2[i] = i;
     }
//     while(true)
//     {
//         test_tensor();
//     }

     for (int i = 0; i < test2.count(); i++) {
         LOG(INFO) << data2[i];
     }





    return 0;
}
