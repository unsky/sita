# author: zhengwenchao@baidu.com
set(Caffe_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/caffe/install)
include_directories(SYSTEM ${Caffe_ROOT}/include)
link_directories(${Caffe_ROOT}/lib)
cuda_include_directories(SYSTEM ${Caffe_ROOT}/include)
cuda_link_directories(${Caffe_ROOT}/lib)
