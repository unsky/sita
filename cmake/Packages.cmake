# author: zhengwenchao@baidu.com

# include this file, you can get the ${include_path}s & %{lib_path}s

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROS.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/GTest.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Boost.cmake)
if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CvBridge.cmake)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Eigen.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenCV.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Protobuf.cmake)
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CCoverage.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Yaml.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenBLAS.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Hdf5.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Leveldb.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Lmdb.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Cuda.cmake)
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Xlog.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Glog.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenGL.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Glew.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Glfw.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/wildmagic5.cmake)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR})
