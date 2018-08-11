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
#include "sita/stuff/shuffle/fisher_yates_shuffler.h"
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
    DataProvider(std::string data_file, std::string label_file, std::vector<Dtype> means, int batch_size, int thread_num,
         bool shuffle): _batch_size(batch_size), _num_thread(thread_num){
    };
    ~DataProvider(){};
    static const int PREFETCH_COUNT = 3;

    inline int num_thread(){
        return _num_thread;
    }
    inline int batch_size(){
        return  _batch_size;
    }

    template <class RandomAccessIterator>
    void shuffle_data(RandomAccessIterator begin, RandomAccessIterator end){
        LOG(INFO) << "shuffling data ...";
        FisherYatesShuffler fy_shuff;
        fy_shuff.shuffle(begin, end);
        LOG(INFO)<<"data is shuffled!!!";
    }

private:
    int _num_thread;
    int _batch_size;

};


}//namespace sita
#endif //SITA_DATA_PROVIDER_H
