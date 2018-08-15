cmake_minimum_required(VERSION 3.5)
project(sita)

set (CMAKE_CXX_STANDARD 11)
##cxx flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2 -m64 -fpermissive -std=c++11 -g -pthread -lboost_system ")

set (CMAKE_VERBOSE_MAKEFILE ON)
set (USE_CUDNN ON)
##cuda flags
set(Cuda_ROOT /home/cs/caffe_third/for_x86/cuda-80)
find_package(CUDA 8.0 REQUIRED)
set (CUDA_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2 -m64 -fpermissive -std=c++11 -g -pthread -lboost_system ")

set(INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/src)
cuda_include_directories(${PROJECT_SOURCE_DIR}/src)
cuda_include_directories(${INCLUDE_DIR})
link_directories(SYSTEM /home/cs/Desktop/cs_work/unsky/sita/src)

link_directories(${Cuda_ROOT}/lib64)
link_directories(${Cuda_ROOT}/lib64/stubs/)

set(GLOG_ROOT /home/cs/caffe_third/for_x86/glog)
set(GLOG_INCLUDE_DIR ${GLOG_ROOT}/include)
set(GLOG_LIBRARY_DIR ${GLOG_ROOT}/lib)
include_directories(${GLOG_INCLUDE_DIR})
link_directories(${GLOG_LIBRARY_DIR})
file(GLOB GLOG_LIBRARIES ${GLOG_LIBRARY_DIR}/*.a)

include_directories(/home/cs/caffe_third/for_x86/protobuf/include)
link_directories(/home/cs/caffe_third/for_x86/protobuf/lib)
execute_process(COMMAND protoc  --cpp_out=${PROJECT_SOURCE_DIR}/src/sita/proto/ ${PROJECT_SOURCE_DIR}/src/sita/protos/sita.proto)


set(SITA_FILES
        src/sita/proto/sita_operators.pb.cc
        src/sita/proto/sita_utils.pb.cc
        main.cpp
        src/sita/memory_control.cpp
        src/sita/tensor.cpp
        src/sita/dlflow/operators/add_op.cpp
        src/sita/workspace.cpp
        src/sita/dlflow/operator.cpp
        src/sita/dlflow/graph.cpp
        src/sita/io_protobuff.cpp
        src/sita/internal_thread.cpp
        src/sita/blocking_queue.cpp
        src/sita/dataprovider/mnist_dataprovider.cpp
        src/sita/dataprovider/dataprovider.cpp
        src/sita/dataprovider/dataset_util/mnist.cpp
        src/sita/stuff/shuffle/fisher_yates_shuffler.cpp
        )


######################c++ and cuda##########################
cuda_add_executable(sita  ${SITA_FILES})

###########################################################


###boost
find_package(Boost 1.55 REQUIRED COMPONENTS system thread filesystem regex)
include_directories(system thread filesystem regex ${Boost_INCLUDE_DIRS})
link_directories(system  thread filesystem regex${Boost_LIBRARIES})
##opencv
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
INCLUDE_DIRECTORIES(core highgui imgproc OpenCV_INCLUDE_DIRS)
link_directories(core highgui imgproc ${OpenCV_LIBRARIES})


target_link_libraries(sita

"/usr/lib/x86_64-linux-gnu/libboost_system.so"
        "/usr/lib/x86_64-linux-gnu/libboost_thread.so"
        "/home/cs/caffe_third/for_x86/protobuf/lib/libprotobuf.so"


        ${OpenCV_LIBRARIES}
"/usr/lib/x86_64-linux-gnu/libglog.so.0"
"/usr/lib/x86_64-linux-gnu/libgflags.so.2"
)




