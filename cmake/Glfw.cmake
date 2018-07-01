# author: yanfeilong@baidu.com

find_package(Threads REQUIRED)

find_library(RT_LIBRARY rt)
mark_as_advanced(RT_LIBRARY)
if (RT_LIBRARY)
   list(APPEND glfw_LIBRARIES "${RT_LIBRARY}")
endif()

find_library(MATH_LIBRARY m)
mark_as_advanced(MATH_LIBRARY)
if (MATH_LIBRARY)
    list(APPEND glfw_LIBRARIES "${MATH_LIBRARY}")
endif()

if (CMAKE_DL_LIBS)
    list(APPEND glfw_LIBRARIES "${CMAKE_DL_LIBS}")
endif()


find_package(X11 REQUIRED)
list(APPEND glfw_LIBRARIES "${X11_X11_LIB}" "${CMAKE_THREAD_LIBS_INIT}")
list(APPEND glfw_LIBRARIES "${X11_Xrandr_LIB}")
list(APPEND glfw_LIBRARIES "${X11_Xinerama_LIB}")

if (X11_xf86vmode_FOUND)
    if (X11_Xxf86vm_LIB)
        list(APPEND glfw_LIBRARIES "${X11_Xxf86vm_LIB}")
    else()
        list(APPEND glfw_LIBRARIES Xxf86vm)
    endif()
endif()

list(APPEND glfw_LIBRARIES "${X11_Xcursor_LIB}")
 
if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
    SET(ADU_3RD_GLFW_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/glfw")
    set(GLFW_DEPENDS libglfw3.a ${glfw_LIBRARIES})
    include_directories(SYSTEM ${ADU_3RD_GLFW_DIR}/include)
    link_directories(${ADU_3RD_GLFW_DIR}/lib)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
    SET(ADU_3RD_GLFW_DIR "${CMAKE_SOURCE_DIR}/../../adu-3rd/glfw")
    set(GLFW_DEPENDS libglfw3.a ${glfw_LIBRARIES})
    include_directories(SYSTEM ${ADU_3RD_GLFW_DIR}/include)
    link_directories(${ADU_3RD_GLFW_DIR}/lib)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
    SET(ADU_3RD_GLFW_DIR "${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/glfw/aarch64")
    set(GLFW_DEPENDS libglfw.so ${glfw_LIBRARIES})
    include_directories(SYSTEM ${ADU_3RD_GLFW_DIR}/include)
    link_directories(${ADU_3RD_GLFW_DIR}/lib)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")

