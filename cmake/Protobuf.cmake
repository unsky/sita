# author: zhengwenchao@baidu.com

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    set(PROTOBUF_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/protobuf)
    set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/include)
    set(PROTOBUF_LIBRARY_DIR ${PROTOBUF_ROOT}/lib/)
    file(GLOB PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARY_DIR}/libproto*)
    set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_ROOT}/bin/protoc)
    set(PROTOC ${PROTOBUF_PROTOC_EXECUTABLE})
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    set(PROTOBUF_ROOT ${CMAKE_SOURCE_DIR}/../../../baidu/adu-3rd/protobuf)
    set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/include)
    set(PROTOBUF_LIBRARY_DIR ${PROTOBUF_ROOT}/lib)
    set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_ROOT}/bin/protoc)
    set(PROTOC ${PROTOBUF_PROTOC_EXECUTABLE})
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    set(PROTOBUF_ROOT ${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/protobuf/aarch64)
    set(PROTOBUF_INCLUDE_DIR ${PROTOBUF_ROOT}/include)
    set(PROTOBUF_LIBRARY_DIR ${PROTOBUF_ROOT}/lib/)
    file(GLOB PROTOBUF_LIBRARIES ${PROTOBUF_LIBRARY_DIR}/libproto*)
    set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_ROOT}/bin/protoc)
    if (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
      set(PROTOBUF_PROTOC_EXECUTABLE /usr/bin/protoc)
    else (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
      set(PROTOBUF_PROTOC_EXECUTABLE ${PROTOBUF_ROOT}/bin/protoc)
    endif (${CROSS_COMPILE} STREQUAL "16.04x86_64-16.04aarch64")
    set(PROTOC ${PROTOBUF_PROTOC_EXECUTABLE})
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")

include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})
link_directories(${PROTOBUF_LIBRARY_DIR})

function(PROTOBUF_GENERATE_CPP SRCS HDRS)

  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif(NOT ARGN)

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOC}
      ARGS --cpp_out  ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_SOURCE_DIR}
                                                  --proto_path ${CMAKE_SOURCE_DIR}
                                                  ${ABS_FIL}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

function(PROTOBUF_GENERATE_CPP_IMPORT1 SRCS HDRS IMPORT_PATH1)

  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif(NOT ARGN)

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOC}
      ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_SOURCE_DIR} 
                                                 --proto_path "${CMAKE_CURRENT_SOURCE_DIR}/${IMPORT_PATH1}" 
                                                 ${ABS_FIL}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

function(PROTOBUF_GENERATE_CPP_IMPORT2 SRCS HDRS IMPORT_PATH1 IMPORT_PATH2)

  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif(NOT ARGN)

  set(${SRCS})
  set(${HDRS})
  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")

    add_custom_command(
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc"
               "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h"
      COMMAND  ${PROTOC}
      ARGS --cpp_out ${CMAKE_CURRENT_BINARY_DIR} --proto_path ${CMAKE_CURRENT_SOURCE_DIR} 
                                                 --proto_path "${CMAKE_CURRENT_SOURCE_DIR}/${IMPORT_PATH1}" 
                                                 --proto_path "${CMAKE_CURRENT_SOURCE_DIR}/${IMPORT_PATH2}" 
                                                 ${ABS_FIL}
      DEPENDS ${ABS_FIL}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

