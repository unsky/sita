//
// Created by unsky on 07/08/18.
//
#include "sita/dataprovider/mnist_dataprovider.h"
namespace  sita{

template <typename Dtype>
MnistDataProvider<Dtype>::MnistDataProvider(std::string data_file, std::string label_file,
        std::vector<Dtype> means, int batch_size, int thread_num, bool shuffle):DataProvider<Dtype>(data_file,
        label_file, means, batch_size, thread_num, shuffle) {
    LOG(INFO) << "loading mnist dataset using "<< thread_num <<" threads ...";
    _threads.resize(thread_num);
    _thread_images.resize(thread_num);
    _thread_labels.resize(thread_num);
    for(int i = 0; i < thread_num; i++){
        _thread_images.clear();
        _thread_labels.clear();
    }

    read_mnist_image(data_file.c_str(), _images);
    read_mnist_label(label_file.c_str(), _labels);

    CHECK_EQ(_images.size(), _labels.size()) << "label size do not equal image size in mnist!!";

    if(shuffle){
        std::vector<std::pair<cv::Mat, double> > shuffled_data;
        for(int i = 0; i < _images.size(); i++){
            shuffled_data.push_back(std::make_pair(_images[i], _labels[i]));
        }
        this->shuffle_data(shuffled_data.begin(), shuffled_data.end());
        _images.clear();
        _labels.clear();
        for(int i = 0; i < shuffled_data.size(); i++){
            _images.push_back(shuffled_data[i].first);
            _labels.push_back(shuffled_data[i].second);
        }
    }

    int num_images_in_one_thread = _images.size()/thread_num - 1;

    LOG(INFO) << num_images_in_one_thread << " data processing in each thread!!!";

    for(int i = 0; i < thread_num; i++){
        int begin_indx = i * num_images_in_one_thread;
        int end_indx = (i + 1) * num_images_in_one_thread;
        for(int j = begin_indx; j < end_indx; j++){
            _thread_images[i].push_back(&_images[j]);
            _thread_labels[i].push_back(&_labels[j]);
        }
    }

    for(int i = 0; i < DataProvider<Dtype>::PREFETCH_COUNT; i++){
        _prefetch[i].data()->reshape(batch_size, _images[0].channels(), _images[0].rows, _images[0].cols);
        _prefetch[i].label()->reshape(1, 1, 1, batch_size);
        _prefetch_free.push(&_prefetch[i]);
    }



    for(int i = 0; i < _threads.size(); i ++){

        _threads[i].init(&_prefetch_free, &_prefetch_full, _thread_images[i], _thread_labels[i], this->means());
        _threads[i].start_internal_thread();
    }
    LOG(INFO) << " end load mnist dataset.";
}

template <typename Dtype>
MnistBatch<Dtype> * MnistDataProvider<Dtype>::fetch_batch(){
    MnistBatch<Dtype>* batch = _prefetch_full.pop("Dataprovider prefetch queue empty!!");
    _prefetch_free.push(batch);
    return batch;
}

template <typename Dtype>
void MnistDataProviderEntry<Dtype>::internal_thread_entry(){
    try
    {
        while (!must_stop())
        {
            MnistBatch<Dtype>* batch = _free->pop("Dataprovider donot prefech batch!!");
            load_batch(batch);
            _full->push(batch);
        }
    }
    catch (boost::thread_interrupted&)
    {
        // Interrupted exception is expected on shutdown
    }

}
template <typename Dtype>
void MnistDataProviderEntry<Dtype>::load_batch(MnistBatch<Dtype>* batch){
    Dtype* data = batch->data()->mutable_cpu_data();
    Dtype* label = batch->label()->mutable_cpu_data();

    for(int b = 0; b < batch->data()->shape(0); b++){

            int offset = batch->data()->get_site_by_coord(b, 0, 0, 0);
            for(int h = 0; h < batch->data()->shape(2); h++){
                for(int w = 0; w < batch->data()->shape(3); w++){
                    data[offset+ h*batch->data()->shape(3) + w] = _images[_index]->at<float>(h, w) - (*_means)[0];
                }
            }
        label[b] = Dtype(*_labels[_index]);
        _index++;
        if(_index >= _images.size()){
            _index = 0;
        }
    }

}

INSTANTIATE_CLASS(MnistDataProvider);

}