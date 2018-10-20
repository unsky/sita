//
// Created by unsky on 02/07/18.
//
#include "sita/graph.h"
namespace sita{

Graph::Graph(std::string model_file){
    CHECK(read_proto_from_txt(model_file.c_str(), &_graph))
            << "Failed to parse SolverParameter file: " << model_file;
}


void Graph::graph_symbol_show(){
    LOG(INFO) << "graph " << _graph.name() << ":";
    LOG(INFO)<<"-------------------------";
    for(int i = 0; i < _graph.operatordef_size(); i ++) {
        LOG(INFO) << "operator name: " << _graph.operatordef(i).name();
        LOG(INFO) << "type: " << _graph.operatordef(i).type();
        std::string input_str = "inputs: ";
        for(int in = 0; in < _graph.operatordef(i).input_size(); in++ ) {
            input_str += (_graph.operatordef(i).input(in) + " ");
        }
        LOG(INFO) << input_str;
        std::string output_str = "outputs: ";
        for(int ou = 0; ou < _graph.operatordef(i).output_size(); ou++){
            output_str += (_graph.operatordef(i).output(ou)+ " ");
        }
        LOG(INFO)<< output_str;
        LOG(INFO)<<"-------------------------";
    }
}


}//namespace

