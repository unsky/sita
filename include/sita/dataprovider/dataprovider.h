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
    Batch(){}
    ~Batch(){};

    virtual std::string product_name(int i) = 0;
    virtual Tensor<Dtype>* product(int i) = 0;
    virtual int product_size() = 0;
    virtual Tensor<Dtype> *data() = 0;
    virtual Tensor<Dtype> *label() = 0;
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
         bool shuffle, std::string type):_means(means), _batch_size(batch_size), _num_thread(thread_num), _type(type){
    };
    ~DataProvider(){};
    static const int PREFETCH_COUNT = 3;
    virtual Batch<Dtype>* fetch_batch()=0;

    inline int num_thread(){
        return _num_thread;
    }
    inline int batch_size(){
        return  _batch_size;
    }
    inline std::vector<Dtype> * means(){
        return  &_means;
    }
    inline std::string type(){
        return _type;
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
    std::vector<Dtype> _means;
    std::string _type;

};


}//namespace sita
#endif //SITA_DATA_PROVIDER_H
