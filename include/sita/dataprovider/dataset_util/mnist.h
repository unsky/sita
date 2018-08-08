//
// Created by unsky on 08/08/18.
//

#ifndef SITA_DATAPROVIDER_MNIST_H
#define SITA_DATAPROVIDER_MNIST_H
#include <vector>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
namespace  sita {

int reverse_int(int i);

void read_mnist_image(char *path, std::vector <cv::Mat>& vec);

void read_mnist_label(char *path,std::vector<double> &labels);


}
#endif //SITA_MNIST_UTIL_H
