//
// Created by unsky on 07/08/18.
//

#ifndef SITA_DATA_PROVIDER_DATA_PROVIDER_H
#define SITA_DATA_PROVIDER_DATA_PROVIDER_H
#include "sita/blocking_queue.h"
#include "sita/internal_thread.h"
#include "sita/tensor.h"
#include <string>
#include <vector>
namespace sita{
template <typename Dtype>
class Batch{
public:
    Batch(): _batch_size(0){}
    Batch(int batch_size): _batch_size(batch_size){}
    ~Batch(){};
    inline int batch_size(){
        return _batch_size;
    }
private:
    int _batch_size;
};

template <typename Dtype>
class DataProviderEntry: public InternalThread{
public:
    DataProviderEntry(){}
    ~DataProviderEntry(){}
private:
    virtual void internal_thread_entry(){};
};


template <typename Dtype>
class DataProvider{
public:
    DataProvider(std::string data_file, std::string label_file, std::vector<Dtype> means, int batch_size, int thread_num){
    };
    ~DataProvider(){};
    static const int PREFETCH_COUNT = 3;

    inline int num_thread(){
        return _num_thread;
    }
    inline int batch_size(){
        return  _batch_size;
    }
    inline std::vector<Dtype> means(){
        return  _means;
    }

private:
    int _num_thread;
    int _batch_size;
    std::vector<Dtype> _means;

};


}//namespace sita
#endif //SITA_DATA_PROVIDER_H
