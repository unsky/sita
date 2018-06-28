cmake_minimum_required(VERSION 3.5)
project(sita)

set (CMAKE_CXX_STANDARD 11)
##cxx flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O2 -m64 -fpermissive -std=c++11 -g -pthread -lboost_system ")

set (CMAKE_VERBOSE_MAKEFILE ON)
set (USE_CUDNN ON)
##cuda flags
set(Cuda_ROOT /home/cs/work/detector/thirdparty/for_x86/cuda-80)
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

set(GLOG_ROOT /home/cs/work/detector/thirdparty/for_x86/glog)
set(GLOG_INCLUDE_DIR ${GLOG_ROOT}/include)
set(GLOG_LIBRARY_DIR ${GLOG_ROOT}/lib)
include_directories(${GLOG_INCLUDE_DIR})
link_directories(${GLOG_LIBRARY_DIR})
file(GLOB GLOG_LIBRARIES ${GLOG_LIBRARY_DIR}/*.a)


set(SITA_FILES
        main.cpp
        src/sita/stuff/memcontrol.cpp
        src/sita/stuff/tensor.cpp
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
"/home/cs/work/baidu/ai-auto/l3-apollo/thirdparty/for_x86/protobuf/lib/libprotobuf.so"
"/usr/lib/x86_64-linux-gnu/libboost_system.so"
        "/usr/lib/x86_64-linux-gnu/libboost_thread.so"

${OpenCV_LIBRARIES}
"/usr/lib/x86_64-linux-gnu/libglog.so.0"
"/usr/lib/x86_64-linux-gnu/libgflags.so.2"
"/home/cs/work/baidu/ai-auto/l3-apollo/thirdparty/for_x86/cuda-80/lib64/libcublas.so"
"/home/cs/work/baidu/ai-auto/l3-apollo/thirdparty/for_x86/cuda-80/lib64/libcudnn.so"
"/home/cs/work/baidu/ai-auto/l3-apollo/thirdparty/for_x86/cuda-80/lib64/libcurand.so"
"/home/cs/work/baidu/ai-auto/l3-apollo/thirdparty/for_x86/OpenBLAS/lib/libopenblas.so"
        "/home/cs/work/baidu/ai-auto/l3-apollo/thirdparty/for_x86/TensorRT-2.1.2/lib/libnvinfer.so.3"
)




