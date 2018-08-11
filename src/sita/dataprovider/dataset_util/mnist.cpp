//
// Created by unsky on 08/08/18.
//

#include <fstream>
#include <vector>
#include <iostream>
#include "sita/dataprovider/dataset_util/mnist.h"
namespace  sita{

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void read_mnist_image(char* path, std::vector<cv::Mat> &vec) {

    std::ifstream is(path, std::ios::in | std::ios::binary);

    if (!is) {
        LOG(FATAL) << "cannot open file: "<< path;
    } else {
        int num_header = 4;
        int header[num_header];
        is.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (is) {
            for (int i = 0; i < num_header; ++i) {
                header[i] = reverse_int(header[i]);
            }
            LOG(INFO) << "read successful!!!";
            LOG(INFO) << "magic number: " << header[0];
            LOG(INFO) << "image number: " << header[1];
            LOG(INFO) << "rows  number: " << header[2];
            LOG(INFO) << "cols  number: " << header[3];
            vec.resize(header[1]);
            unsigned char temp = 0;
            for (int i = 0; i < header[1]; ++i) {
                cv::Mat image = cv::Mat::zeros(header[2], header[3], CV_64FC1);

                for (int r = 0; r < header[2]; ++r) {
                    for (int c = 0; c < header[3]; ++c) {
                        is.read((char*) &temp, sizeof(temp));
                        image.at < float > (r, c) = (float) temp;
                    }
                }
                vec[i] = image;
            }
        }
    }
    is.close();
}

void read_mnist_label(char* path, std::vector<double> &labels){

    std::ifstream is(path, std::ios::in | std::ios::binary);

    if (!is) {
        LOG(FATAL)<< "cannot open file: "<< path;
    } else {
        int num_header = 2;
        int header[num_header];
        is.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (is) {
            for (int i = 0; i < num_header; ++i) {
                header[i] = reverse_int(header[i]);
            }

            LOG(INFO) << "read successful!!!";
            LOG(INFO) << "magic number: " << header[0];
            LOG(INFO) << "label number: " << header[1];

            labels.resize(header[1]);
            unsigned char temp = 0;
            int size = sizeof(temp);
            for (int i = 0; i < header[1]; ++i) {
                is.read((char*) &temp,size);
                labels[i]=(double)temp;
            }
        }
    }
    is.close();
}

}//namespace