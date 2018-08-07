//
// Created by unsky on 07/08/18.
//
#include "sita/dataprovider/mnist_dataprovider.h"
namespace  sita{
    template <typename Dtype>
    MnistBatch<Dtype> * MnistDataProvider<Dtype>::fetch_batch(){
        MnistBatch<Dtype>* batch = _prefetch_full.pop("Dataprovider prefetch queue empty");

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