# author: zhengwenchao@baidu.com

set(LMDB_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/lmdb)
include_directories(SYSTEM ${LMDB_ROOT}/include)
link_directories(${LMDB_ROOT}/lib)

