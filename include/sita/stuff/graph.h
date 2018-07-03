//
// Created by unsky on 02/07/18.
//

#ifndef SITA_STUFF_GRAPH_H
#define SITA_STUFF_GRAPH_H
#include <vector>
#include <glog/logging.h>
#include "macros.h"
#include "operator.h"
#include "sita/protos/sita.pb.h"
namespace sita{

class Graph{
public:
    Graph(std::string name);
    ~Graph(){};
    inline const std::string graph_name() const{
        return _grap_sym.name();
    }

    void append(std::string op_type, std::string name, std::vector<std::string> inputs, std::vector<std::string> outputs);
    void graph_symbol_show();
    void get_graph_instance();


private:
    GraphSym _grap_sym;

    std::vector<Operator<float>> _ops;
    DISABLE_COPY_AND_ASSIGN(Graph);
};

}//namespace


#endif //SITA_STUFF_GRAPH_H
