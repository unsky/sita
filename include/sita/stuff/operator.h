//
// Created by unsky on 03/07/18.
//

#ifndef SITA_STUFF_OPERATORS_H
#define SITA_STUFF_OPERATORS_H
#include <vector>
#include <string>
#include "macros.h"
#include "workspace.h"
#include "sita/protos/sita.pb.h"
namespace sita{

template <typename Dtype>
class GlobalWorkSpace;

template <typename Dtype>
class Operator{
public:
    Operator(const OperatorDef&, GlobalWorkSpace<Dtype> *gws):_gws(gws){}
    ~Operator(){}
    void  init();
    virtual void forward(int num){};
    virtual void backward(){};
protected:
    GlobalWorkSpace<Dtype> *_gws;
};

}//namespace


#endif //SITA_STUFF_OPERATORS_H
