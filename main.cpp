//
// Created by unsky on 28/06/18.
//
#include "sita/stuff/common.h"
#include "sita/stuff/memcontrol.h"
#include "sita/stuff/tensor.h"
#include <glog/logging.h>
int main(int argc, char** argv) {
     const int a = 2;
     std::vector<int > aa;
     aa.push_back(10);
    aa.push_back(12);
    sita::Tensor<float> *test = new sita::Tensor<float>(aa);
    LOG(INFO) << test->shape_string();
    const  float *data = test->cpu_data();
    for(int i = 0; i < test->count(); i++) {
        LOG(INFO) << data[i];
    }


    return 0;
}
