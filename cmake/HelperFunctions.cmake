# Copyright 2016 Baidu Inc. All Rights Reserved.
# @author: Ziqiang Huan (huanziqiang@baidu.com)
# @file: HelperFunctions.cmake
# @brief: some helper functions for your cmake

# @brief: install file or directory (it can deal with symlink in it)
#         actually, it is a strong version for install
# @usage:
#       smart_install_one(data ${CUR_BINARY_NAME})
# @note: uniformly you should use smart_install instead
function(smart_install_one arg)
    get_filename_component(real_argv0 "${ARGV0}" REALPATH)
    get_filename_component(real_argv1 "${ARGV1}" REALPATH)

    if(IS_DIRECTORY "${real_argv0}")
        file(GLOB dir_files ${real_argv0}/*)
        get_filename_component(last_dir "${real_argv0}" NAME)
        foreach(item_file ${dir_files})
            smart_install_one(${item_file} ${real_argv1}/${last_dir})
        endforeach()
    else()
        install(FILES ${real_argv0}
                DESTINATION ${real_argv1})
    endif()
endfunction()

# @brief: install file or directory (it can deal with symlink in it)
#         actually, it is a strong version for install
# @usage:
#       smart_install(data conf ${CUR_BINARY_NAME})
function(smart_install arg)
    set(SRCS "")
    set(DIST "")
    math(EXPR max_arg_index ${ARGC}-1)
    math(EXPR cnt_loop ${ARGC}-2)
    list(GET ARGV ${max_arg_index} DIST)
    foreach(idx RANGE ${cnt_loop})
        list(GET ARGV ${idx} idx_arg)
        list(APPEND SRCS ${idx_arg})
        smart_install_one(${idx_arg} ${DIST})
    endforeach()
    message(STATUS "SRCS: ${SRCS}, DIST: ${DIST}")
endfunction()
