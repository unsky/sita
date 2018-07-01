# author: zhengwenchao@baidu.com

set(OpenBLAS_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/OpenBLAS)
include_directories(SYSTEM ${OpenBLAS_ROOT}/include)
link_directories(${OpenBLAS_ROOT}/lib)

