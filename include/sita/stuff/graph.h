//
// Created by unsky on 02/07/18.
//

#ifndef SITA_STUFF_GRAPH_H
#define SITA_STUFF_GRAPH_H
#include <vector>
#include <glog/logging.h>
#include "macros.h"
#include "sita/protos/sita.pb.h"
namespace sita{

class Graph{
public:
    Graph(std::string name);
    ~Graph(){};

    void append(std::string op_type, std::string name, std::vector<std::string> inputs, std::vector<std::string> outputs);
    void graph_symbol_show();
    GraphSym * graph_sym(){
        return &_graph_sym;}

private:
    GraphSym _graph_sym;

};

}//namespace


#endif //SITA_STUFF_GRAPH_H
