//
// Created by unsky on 07/08/18.
//

#ifndef SITA_DATA_PROVIDER_MNIST_DATAPROVIDER_H
#define SITA_DATA_PROVIDER_MNIST_DATAPROVIDER_H
#include "dataprovider.h"
#include "dataset_util/mnist.h"
namespace sita {

template <typename Dtype>
class MnistBatch : public Batch<Dtype> {
public:
    MnistBatch(): Batch<Dtype>(0){}
    MnistBatch(int batch_size): Batch<Dtype>(batch_size){}
    ~MnistBatch(){}
    inline Tensor<Dtype> *data(){
        return &_data;
    }
    inline Tensor<Dtype> *label(){
        return &_label;
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
    inline void init(BlockingQueue<MnistBatch<Dtype>*> *free, BlockingQueue<MnistBatch<Dtype>*> *full,
                     std::vector<cv::Mat *>  images, std::vector<double *> labels, std::vector<Dtype *>  means){
        _free = free;
        _full = full;
        _images = images;
        _labels = labels;
        _means = means;
    }
    void load_batch(MnistBatch<Dtype> * batch);
private:
    virtual void internal_thread_entry();
    BlockingQueue<MnistBatch<Dtype>*> * _free;
    BlockingQueue<MnistBatch<Dtype>*> * _full;
    std::vector<cv::Mat* > _images;
    std::vector<double * >  _labels;
    std::vector<Dtype *> _means;
    int _index = 0;
};

template <typename Dtype>
class MnistDataProvider: public DataProvider<Dtype>{
public:
    MnistDataProvider(std::string data_file, std::string label_file,
      std::vector<Dtype> means, int batch_size, int thread_num,bool shuffle);
    ~MnistDataProvider(){
        for(int i = 0; i < _threads.size(); i++){
            _threads[i].stop_internal_thread();//join main processor
        }

    }

    MnistBatch<Dtype> * fetch_batch();
private:
    std::vector<MnistDataProviderEntry<Dtype>> _threads;
    MnistBatch<Dtype> _prefetch[DataProvider<Dtype>::PREFETCH_COUNT];
    BlockingQueue<MnistBatch<Dtype>*> _prefetch_free;
    BlockingQueue<MnistBatch<Dtype>*> _prefetch_full;
    std::vector<cv::Mat> _images;
    std::vector<double> _labels;
    std::vector<Dtype> _means;
    std::vector<std::vector<cv::Mat *> > _thread_images;
    std::vector<std::vector<double *> > _thread_labels;
};
}//namespace sita

#endif //SITA_MNIST_DATAPROVIDER_H
