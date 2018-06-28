# author: zhengwenchao@baidu.com

# Perception Version Info, will be compiled in binary.
set(build_version "1.0.0.0")

######################## enviroment      #####################################################
set (CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build/install CACHE STRING "" FORCE)
                                                                # set default install dir
set (CMAKE_CXX_FLAGS "-std=c++11 -g -fPIC -O2 -DNDEBUG -fopenmp -Wall" CACHE STRING "" FORCE)
                                                                # -O2 need to be added back
                                                                # DO NOT remove -g flags,
                                                                # we should reserve symbol
                                                                # info to debug coredump  file.
set (COV_HOME ${CMAKE_SOURCE_DIR}/lib/tools/ccover-8.9/)
if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    set (CMAKE_C_COMPILER /usr/bin/gcc-5 CACHE STRING "" FORCE)
    set (CMAKE_CXX_COMPILER /usr/bin/g++-5 CACHE STRING "" FORCE)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    set (CMAKE_CXX_COMPILER ${COV_HOME}/bin/g++ CACHE STRING "" FORCE)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    if (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
        set (CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc CACHE STRING "" FORCE)
        set (CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++ CACHE STRING "" FORCE)
    else (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
        set (CMAKE_C_COMPILER /usr/bin/gcc-5 CACHE STRING "" FORCE)
        set (CMAKE_CXX_COMPILER /usr/bin/g++-5 CACHE STRING "" FORCE)
    endif (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")

######################## option switches #####################################################
if ("$ENV{CORE_TYPE}" STREQUAL "GPU")
    SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-cudart=static")
    option(CPU_ONLY  "Build caffe without CUDA support" OFF)
else ("$ENV{CORE_TYPE}" STREQUAL "GPU")
    option(CPU_ONLY  "Build caffe without CUDA support" ON)
    add_definitions(-DCPU_ONLY)
endif ("$ENV{CORE_TYPE}" STREQUAL "GPU")

set(BUILD_SHARED_LIBS         OFF)       # add_library(...) -> .os/.a
option(test "Build all tests." ON)       # Makes boolean 'test' available.

option(USING_OPENMP "Whether using openmp." ON)
option(DISABLE_PERF "Whether disable perf time log." OFF)
option(NDEBUG "Whether release version." OFF)

if ("$ENV{QUICK_CALIB}" STREQUAL "1")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DQUICK_CALIBRATION")
endif ("$ENV{QUICK_CALIB}" STREQUAL "1") 
