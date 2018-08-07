//
// Created by unsky on 07/08/18.
//

#ifndef SITA_DATA_PROVIDER_MNIST_DATAPROVIDER_H
#define SITA_DATA_PROVIDER_MNIST_DATAPROVIDER_H
#include "dataprovider.h"
namespace sita {

    template <typename Dtype>
    class MnistBatch : public Batch<Dtype> {
    public:
        MnistBatch(): Batch<Dtype>(0){}
        MnistBatch(int batch_size): Batch<Dtype>(batch_size){}
        ~MnistBatch(){}
        inline Tensor<Dtype> data(){
            return _data;
        }
        inline Tensor<Dtype> label(){
            return _label;
        }
    private:
        Tensor<Dtype> _data;
        Tensor<Dtype> _label;
    };


    template <typename Dtype>
    class MnistDataProviderEntry: public DataProviderEntry<Dtype>{
    public:
        MnistDataProviderEntry():DataProviderEntry<Dtype>(){}
        ~MnistDataProviderEntry(){}
        inline void init(BlockingQueue<MnistBatch<Dtype>*> *free, BlockingQueue<MnistBatch<Dtype>*> *full){
            _free = free;
            _full = full;
        }
    private:
        virtual void internal_thread_entry();
        BlockingQueue<MnistBatch<Dtype>*> * _free;
        BlockingQueue<MnistBatch<Dtype>*> * _full;
    };

    template <typename Dtype>
    class MnistDataProvider: public DataProvider<Dtype>{
    public:
        MnistDataProvider(std::string data_file, std::string label_file,
          std::vector<Dtype> means, int batch_size, int thread_num) :DataProvider<Dtype>(data_file,
          label_file, means, batch_size, thread_num){
            _threads.resize(thread_num);
            for(int i = 0; i < DataProvider<Dtype>::PREFETCH_COUNT; i++){
                _prefetch[i].data().reshape(10,10,10,10);
                _prefetch[i].label().reshape(10,10,10,1);
                _prefetch_free.push(&_prefetch[i]);
            }
            for(int i = 0; i < _threads.size(); i ++){
                _threads[i].init(&_prefetch_free, &_prefetch_full);
                _threads[i].start_internal_thread();
            }
        }
        ~MnistDataProvider(){}

        MnistBatch<Dtype> * fetch_batch();

    private:
        std::vector<MnistDataProviderEntry<Dtype>> _threads;
        MnistBatch<Dtype> _prefetch[DataProvider<Dtype>::PREFETCH_COUNT];
        BlockingQueue<MnistBatch<Dtype>*> _prefetch_free;
        BlockingQueue<MnistBatch<Dtype>*> _prefetch_full;
    };
}//namespace sita

#endif //SITA_MNIST_DATAPROVIDER_H
