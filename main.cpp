//
// Created by unsky on 28/06/18.
//
#include "sita/stuff/macros.h"
#include "sita/stuff/memory_control.h"
#include "sita/stuff/tensor.h"
#include "sita/stuff/workspace.h"
#include "sita/stuff/graph.h"
#include <glog/logging.h>

int main(int argc, char** argv) {
    sita::GlobalWorkSpace<float > gws;
    gws.device_query();
    gws.set_device(0);

    std::string model_file = "../test.prototxt";
    sita::Graph graph(model_file);

    gws.build_graph(&graph);
    gws.global_init();
    int k = 0;
    while(k!=1) {
        k++;
      //  LOG(INFO) << gws.temp_tensor_memory_size();
        gws.train();
    }
    //workspace test


    return 0;
}
