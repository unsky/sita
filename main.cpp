//
// Created by unsky on 28/06/18.
//
#include "sita/macros.h"
#include "sita/memory_control.h"
#include "sita/tensor.h"
#include "sita/workspace.h"
#include "sita/dlflow/graph.h"
#include <glog/logging.h>
#include "sita/dataprovider/mnist_dataprovider.h"
int main(int argc, char** argv) {
    sita::GlobalWorkSpace<float > gws;
    gws.device_query();
    gws.set_device(0);

    std::string model_file = "../test.prototxt";
    sita::Graph graph(model_file);

    gws.build_graph(&graph);
    gws.global_init();
    std::vector<float> means;
    means.push_back(float(0));
    means.push_back(float(5));
    means.push_back(float(10));
    sita::MnistDataProvider<float > mnistdp("../data/mnist/train-images-idx3-ubyte",
                "../data/mnist/train-labels-idx1-ubyte",means,10,2);

    int k = 0;

    while(k!=100000) {
        k++;
        mnistdp.fetch_batch();
      //  LOG(INFO) << gws.temp_tensor_memory_size();
        gws.train();
    }
    //workspace test


    return 0;
}
