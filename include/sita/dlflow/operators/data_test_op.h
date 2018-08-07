//
// Created by cs on 02/08/18.
//

#ifndef SITA_STUFF_DATA_TEST_OP_H
#define SITA_STUFF_DATA_TEST_OP_H
#include <string>
#include <vector>
#include "sita/stuff/operator.h"
#include "sita/stuff/registry.h"
namespace sita{

    template<typename Dtype>
    class DataTestOp: public Operator<Dtype>{
    public:
        DataTestOp(const OperatorDef& opdef, GlobalWorkSpace<Dtype> *gws):Operator<Dtype>(opdef,gws){
            if(_has_param){
                _filler = opdef.param.filler;
            }
            _data_test_op_param = opdef.param.data_test_op_param;
        }
        ~DataTestOp(){};
        void init();
        void forward();
        void backward();
        bool inline has_param(){ return _has_param;}

    protected:
        bool _has_param = false;
        Filler _filler;
        AddOpParameter _data_op_param;
        std::vector<std::string> _inputs;
        std::vector<std::string> _outputs;

    };
}//namespace



#endif //SITA_DATA_TEST_OP_H
