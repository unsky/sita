####################### third party libs env ##################################################

add_definitions(-DUSE_OPENCV)

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    add_definitions(-DUSE_AARCH64)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
# # set ADU_common libs
# set(ADU_COMMON_ROOT "${CMAKE_SOURCE_DIR}/../../adu/common")
# set(ADU_COMMON_INCLUDE_DIR "${ADU_COMMON_ROOT}/include")
# set(ADU_COMMON_LINK_DIR "${ADU_COMMON_ROOT}/lib")
# include_directories(SYSTEM ${ADU_COMMON_INCLUDE_DIR})
# link_directories(${ADU_COMMON_LINK_DIR})
# 
# # Set idl-car/messages dirs.
# SET(IDL_CAR_MESSAGES_LIB_DIR "${CMAKE_SOURCE_DIR}/../../idl-car/messages")
# include_directories(SYSTEM ${IDL_CAR_MESSAGES_LIB_DIR}/obstacle_msgs/devel/include)
# 
# # Set pnc-common dirs.
# SET(PNC_COMMON_DIR "${CMAKE_SOURCE_DIR}/../pnc-common")
# include_directories(SYSTEM ${PNC_COMMON_DIR}/include)
# link_directories(${PNC_COMMON_DIR}/build)



# Set idl-car/messages dirs.
SET(IDL_CAR_MESSAGES_LIB_DIR "${CMAKE_SOURCE_DIR}/3rd_lib.14/obstacle_msgs")
include_directories(SYSTEM ${IDL_CAR_MESSAGES_LIB_DIR}/include)

## Set caffe dirs
#if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
#    set(Caffe_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/caffe/install)
#    include_directories(SYSTEM ${Caffe_ROOT}/include)
#    link_directories(${Caffe_ROOT}/lib)
#    cuda_include_directories(SYSTEM ${Caffe_ROOT}/include)
#elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
#    set(Caffe_ROOT ${CMAKE_SOURCE_DIR}/../../adu-3rd/caffe-output)
#    include_directories(SYSTEM ${Caffe_ROOT}/include)
#    link_directories(${Caffe_ROOT}/lib)
#elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
#    set(Caffe_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/caffe-output/aarch64)
#    include_directories(SYSTEM ${Caffe_ROOT}/include)
#    link_directories(${Caffe_ROOT}/lib)
#endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")

#################################################################################################

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    SET(ANAKIN ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/anakin/)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    SET(ANAKIN ${CMAKE_SOURCE_DIR}/../../adu-lab/anakin-output)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    SET(ANAKIN ${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/anakin/aarch64)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
include_directories(SYSTEM ${ANAKIN}/include/)
link_directories(${ANAKIN}/lib)

# Set idl-car common-i-lib dirs.
if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    SET(IDL_CAR_COMMON_I_LIB_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/")
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    SET(IDL_CAR_COMMON_I_LIB_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/")
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    SET(IDL_CAR_COMMON_I_LIB_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/")
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
include_directories(SYSTEM ${IDL_CAR_COMMON_I_LIB_DIR})
link_directories(${IDL_CAR_COMMON_I_LIB_DIR})

# include build dir, pb.h pb.cc generated in build dir.
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/build)

set(CUDA_LIBS libnppi_static.a libnppc_static.a libcudart_static.a libcurand_static.a
    libcudnn_static.a libculibos.a rt libcublas_static.a boost_python python2.7)

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    set(CAFFE_DEPENDS ${CUDA_LIBS} ${OpenCV_LIBS} ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES} libopenblas.a libhdf5_hl.a libhdf5.a libglog.a gflags dl cuda ${GLOG_LIBRARIES} )
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    set(CAFFE_DEPENDS libproto.a liblmdb.a libleveldb.a libopenblas.a libcublas.so ${CUDA_LIBS} dl ${OpenCV_LIBS} z cuda ${GLOG_LIBRARIES} gflags)
    set(CAFFE_DEPENDS  ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES} libcblas.so ${GLOG_LIBRARIES} gflags ${OpenCV_LIBS} rt ${CUDA_LIBS} dl cudnn)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    if (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
        set(CAFFE_DEPENDS  ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES} $ENV{ARM64_ROOT}/usr/lib/atlas-base/libcblas.so ${GLOG_LIBRARIES} gflags ${OpenCV_LIBS} rt ${CUDA_LIBS} dl cudnn)
    else (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
        set(CAFFE_DEPENDS  ${PROTOBUF_LIBRARIES} ${Boost_LIBRARIES} libcblas.so ${GLOG_LIBRARIES} gflags ${OpenCV_LIBS} rt ${CUDA_LIBS} dl cudnn)
    endif (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")

set(Caffe_LINK -Wl,--whole-archive libcaffe.so  libcaffe.a -Wl,--no-whole-archive)
set(Anakin_LINK lib_anakin.so)

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    set(WildMagic5_LINK libWm5Core.a libWm5Mathematics.a)
else ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    set(WildMagic5_LINK libWm5Core.so libWm5Mathematics.so)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")

#set(Tensorrt_LINK libnvinfer.a libnvinfer_plugin.a libnvcaffe_parser.a)

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    # driveworks
    include_directories(/usr/share/visionworks/sources/3rdparty/nvmedia)
    include_directories(/usr/local/cuda/include/)
    include_directories(/usr/local/driveworks/include/
                        /usr/local/driveworks/include/dw/
                        /usr/local/driveworks/include/core/
                        /usr/local/driveworks/include/dw/image/
                        /usr/local/driveworks/include/dw/sensors/)
    link_directories(/usr/local/driveworks/lib/)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")

