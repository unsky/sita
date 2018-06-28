# author: chengyiyuan@baidu.com


find_package(OpenGL REQUIRED)

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    SET(ADU_3RD_GLEW_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/glew")
    set(GLEW_DEPENDS libGLEW.a) 
    include_directories(SYSTEM ${ADU_3RD_GLEW_DIR}/include)
    link_directories(${ADU_3RD_GLEW_DIR}/lib)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    SET(ADU_3RD_GLEW_DIR "${CMAKE_SOURCE_DIR}/../../adu-3rd/glew")
    set(GLEW_DEPENDS libGLEW.a) 
    include_directories(SYSTEM ${ADU_3RD_GLEW_DIR}/include)
    link_directories(${ADU_3RD_GLEW_DIR}/lib)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    SET(ADU_3RD_GLEW_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/glew/aarch64")
    set(GLEW_DEPENDS libGLEW.a) 
    include_directories(SYSTEM ${ADU_3RD_GLEW_DIR}/include)
    link_directories(${ADU_3RD_GLEW_DIR}/lib)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
 

