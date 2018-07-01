# author: zhengwenchao@baidu.com

set(LEVELDB_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/leveldb)
include_directories(SYSTEM ${LEVELDB_ROOT}/include)
link_directories(${LEVELDB_ROOT}/lib)

