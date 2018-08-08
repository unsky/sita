//
// Created by unsky on 07/08/18.
//
#include "sita/dataprovider/mnist_dataprovider.h"
namespace  sita{



template <typename Dtype>
MnistDataProvider<Dtype>::MnistDataProvider(std::string data_file, std::string label_file,
        std::vector<Dtype> means, int batch_size, int thread_num):DataProvider<Dtype>(data_file,
        label_file, means, batch_size, thread_num) {
    _threads.resize(thread_num);
    read_mnist_image(data_file.c_str(), _images);
    read_mnist_label(label_file.c_str(), _labels);

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

template <typename Dtype>
MnistBatch<Dtype> * MnistDataProvider<Dtype>::fetch_batch(){
    MnistBatch<Dtype>* batch = _prefetch_full.pop("Dataprovider prefetch queue empty!!");

    _prefetch_free.push(batch);
}

template <typename Dtype>
void MnistDataProviderEntry<Dtype>::internal_thread_entry(){
    try
    {
        while (!must_stop())
        {
            MnistBatch<Dtype>* batch = _free->pop("Dataprovider donot prefech batch!!");
            //load_bacth
             //      LOG(INFO)<<"--"<<_free->size();
            _full->push(batch);
        }
    }
    catch (boost::thread_interrupted&)
    {
        // Interrupted exception is expected on shutdown
    }

}

INSTANTIATE_CLASS(MnistDataProvider);

}