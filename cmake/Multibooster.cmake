# author: zhengwenchao@baidu.com

set(MULTIBOOSTER_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/multibooster/)

include_directories(SYSTEM ${MULTIBOOSTER_ROOT}/include/)
include_directories(SYSTEM ${MULTIBOOSTER_ROOT}/include/lib) #FIXME: Is the directory name wrong???
link_directories(${MULTIBOOSTER_ROOT}/lib/)
