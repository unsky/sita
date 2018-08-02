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
    int k = 0;

    sita::Graph graph("lenet");

    std::vector<std::string> inputs;
    inputs.push_back("images");
    inputs.push_back("data");
    std::vector<std::string> outputs;
    outputs.push_back("add_res");
    sita::SitaParameter data;
    data.add_op_param.stride_w = 5;
    data.add_op_param.stride_h = 6;
    data.filler.type = "gauss";
    graph.append("AddOp", "data", inputs, outputs, data);


    inputs.clear();
    inputs.push_back("add_res");
    outputs.clear();
    outputs.push_back("loss");
    sita::SitaParameter add1;
    add1.add_op_param.stride_w = 10;
    graph.append("AddOp", "add1", inputs, outputs, add1);

    gws.build_graph(&graph);
    gws.global_init();

    while(k!=1) {
        k++;
      //  LOG(INFO) << gws.temp_tensor_memory_size();
        gws.train();


    }
    //workspace test


    return 0;
}
