# author: zhengwenchao@baidu.com

set(HDF5_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/hdf5)
include_directories(SYSTEM ${HDF5_ROOT}/include)
link_directories(${HDF5_ROOT}/lib)

