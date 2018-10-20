//
// Created by unsky on 02/07/18.
//

#ifndef SITA_METRIC_METRIC_H
#define SITA_METRIC_METRIC_H
#include <vector>
#include <glog/logging.h>
#include <string>
#include "sita/macros.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include "sita/proto/sita.h"
#include "sita/io_protobuff.h"
namespace sita{

class Metric{
public:
    Metric(std::string model_file);
    ~Metric(){};
    void graph_symbol_show();
    GraphParameter * graph_sym(){
        return &_graph;}

private:
    GraphParameter _graph;

};

}//namespace


#endif //SITA_STUFF_GRAPH_H
