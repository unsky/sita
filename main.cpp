//
// Created by unsky on 28/06/18.
//
#include "sita/stuff/macros.h"
#include "sita/stuff/memory_control.h"
#include "sita/stuff/tensor.h"
#include <glog/logging.h>


int test_tensor() {
    sita::Tensor<float> t(1, 1, 1, 4);

}
int main(int argc, char** argv) {
    while(true) {
        std::vector<int> shape;

        shape.push_back(100);
        shape.push_back(100);
        shape.push_back(100);
        shape.push_back(200);

        boost::shared_ptr <sita::Tensor<float>> test1(new sita::Tensor<float>(shape));

        LOG(INFO) << test1->shape_string() << INT_MAX;

        const float *data1 = test1->cpu_data();

//    for (int i = 0; i < test1->count(); i++) {
//        LOG(INFO) << data1[i];
//    }



        sita::Tensor<float> test2(1, 1, 1, 3);
        LOG(INFO) << test2.shape_string();
        float *data2 = test2.mutable_cpu_data();

        test2.gpu_data();

//    for (int i = 0; i < test2.count(); i++) {
//        LOG(INFO) << data2[i];
//        data2[i] = i;
//    }

        test2.reshape(800, 200, 1, 1);
        LOG(INFO) << test2.shape_string() << test2.count();

        float *data3 = test2.mutable_cpu_data();
        for (int i = 0; i < test2.count(); i++) {
            //  LOG(INFO) << data3[i];
        }

        sita::Tensor<float> t1;
        t1.copy_from(test2);
        LOG(INFO) << "t1: " << t1.shape_string() << t1.count();
        t1.set_data_zero();
        for (int i = 0; i < t1.count(); i++) {
            //   LOG(INFO) << t1.cpu_data()[i];
        }
        LOG(INFO) << t1.get_site_by_coord(0, 1, 0, 0);

        sita::Tensor<float> t2(1, 100, 1000, 1000);
        t2.gpu_data();
        LOG(INFO) << t2.get_site_by_coord(0, 60, 50, 100);


    }

    return 0;
}
