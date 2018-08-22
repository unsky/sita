//
// Created by unsky on 28/06/18.
//
#include "sita/macros.h"
#include "sita/memory_control.h"
#include "sita/tensor.h"
#include "sita/workspace.h"
#include "sita/dlflow/graph.h"
#include <glog/logging.h>
#include "sita/dataprovider/mnist_dataprovider.h"
int main(int argc, char** argv) {
    sita::GlobalWorkSpace<float > gws;
    gws.device_query();
    gws.set_device(0);

    std::string model_file = "../test.prototxt";
    sita::Graph graph(model_file);


    std::vector<float> means;
    means.push_back(float(0));
    sita::MnistDataProvider<float > mnistdp("../data/mnist/train-images-idx3-ubyte",
                "../data/mnist/train-labels-idx1-ubyte", means, 10, 10,true);

    gws.global_init(&graph, &mnistdp);

    int k = 0;

    while(k != 10){
        gws.train();
        k++;
    }
    gws.temp_memory();
    gws.flow_memory();
    gws.param_memory();
    return 0;
}



//
//k++;
//sita::TempTensor<float> t = gws.new_tensor();
//sita::Tensor<float> * a = t.tensor;
//a->reshape(9,4,5,6);
//// gws.free_tensor(t);
//
//
//sita::MnistBatch<float> * batch = mnistdp.fetch_batch();
////   LOG(INFO)<<batch->label()->cpu_data()[0];
//
//const float *blob_data = batch->data()->cpu_data();
//cv::Mat cv_img_original(batch->data()->shape(2), batch->data()->shape(3), CV_32FC1);
//for(int b = 0; b<batch->data()->shape(0); b++){
//int offset = batch->data()->get_site_by_coord(b, 0, 0, 0);
//for(int h = 0; h < batch->data()->shape(2); h++){
//for(int w = 0; w < batch->data()->shape(3); w++){
//float value = blob_data[offset + h*batch->data()->shape(3) + w];
//cv_img_original.at<float>(h, w) = value;
////  std::cout<<value;
//}
////  std::cout<<std::endl;
//}
//cv::imwrite("vis/" + std::to_string(k)+"_"+ std::to_string(b)+"__"+std::to_string(int(batch->label()->cpu_data()[b]))+ ".jpg", cv_img_original);
//
//}
