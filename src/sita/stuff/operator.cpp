//
// Created by unsky on 27/06/18.
//
#include "sita/stuff/operator.h"
sita{

template<typename Dtype>
void Operator<Dtype>::init(){};



INSTANTIATE_CLASS(Operator);
DEFINE_REGISTRY(OperatorRegistry, Operator, const OperatorDef&, Workspace*);

}//namespace;