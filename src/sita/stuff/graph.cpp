//
// Created by unsky on 02/07/18.
//
#include "sita/stuff/graph.h"
namespace sita{

Graph::Graph(std::string name){
    _graph_sym.graph_name = name;
}

void Graph::append(std::string op_type, std::string name, std::vector<std::string> inputs, std::vector<std::string> outputs, SitaParameter param){

    for(int i = 0; i < _graph_sym.ops.size(); i++){
        for(int j = 0; j < inputs.size(); j ++){
            for(int i_out = 0; i_out < _graph_sym.ops[i].outputs.size(); i_out ++){
                if(outputs[j] == _graph_sym.ops[i].outputs[i_out])
                    LOG(FATAL) << "output: " <<  _graph_sym.ops[i].outputs[i_out]
                               << " in: " << name << " is confilt with: " << _graph_sym.ops[i].name;
            }
        }
    }
    OperatorDef op;
    op.name  = name;
    op.type = op_type;
    op.param = param;
    for(int i = 0; i < inputs.size(); i ++){
        op.inputs.push_back(inputs[i]);
    }
    for(int i = 0; i < outputs.size(); i ++){
        op.outputs.push_back(outputs[i]);
    }
    _graph_sym.ops.push_back(op);
}

void Graph::graph_symbol_show(){
    LOG(INFO) << "graph " << _graph_sym.graph_name << ":";
    LOG(INFO)<<"-------------------------";
    for(int i = 0; i < _graph_sym.ops.size(); i ++) {
        LOG(INFO) << "op name: " << _graph_sym.ops[i].name;
        LOG(INFO) << "type: " << _graph_sym.ops[i].type;
        LOG(INFO) << "inputs:";
        for(int in = 0; in < _graph_sym.ops[i].inputs.size(); in++ ) {
            LOG(INFO) << _graph_sym.ops[i].inputs[in];
        }
        LOG(INFO) << "outputs:";
        for(int ou = 0; ou < _graph_sym.ops[i].outputs.size(); ou++){
            LOG(INFO) << _graph_sym.ops[i].outputs[ou] << " ";
        }
        LOG(INFO)<<"-------------------------";
    }
}


}//namespace

