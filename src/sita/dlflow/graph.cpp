//
// Created by unsky on 02/07/18.
//
#include "sita/dlflow/graph.h"
namespace sita{

Graph::Graph(std::string model_file){
    CHECK(ReadProtoFromTextFile(model_file.c_str(), &_graph))
            << "Failed to parse SolverParameter file: " << model_file;
}


void Graph::graph_symbol_show(){
    LOG(INFO) << "graph " << _graph.name() << ":";
    LOG(INFO)<<"-------------------------";
    for(int i = 0; i < _graph.operatordef_size(); i ++) {
        LOG(INFO) << "operator name: " << _graph.operatordef(i).name();
        LOG(INFO) << "type: " << _graph.operatordef(i).type();
        LOG(INFO) << "inputs:";
        for(int in = 0; in < _graph.operatordef(i).input_size(); in++ ) {
            LOG(INFO) << _graph.operatordef(i).input(in);
        }
        LOG(INFO) << "outputs:";
        for(int ou = 0; ou < _graph.operatordef(i).output_size(); ou++){
            LOG(INFO) << _graph.operatordef(i).output(ou) << " ";
        }
        LOG(INFO)<<"-------------------------";
    }
}


}//namespace

