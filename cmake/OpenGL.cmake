# author: yanfeilong@baidu.com

find_package(OpenGL REQUIRED)
include_directories(SYSTEM ${OPENGL_INCLUDE_DIR})
# usage: target_link_libraries(${ProjNameApp} ${OPENGL_LIBRARIES})

#find_package(GLUT REQUIRED)
#include_directories(SYSTEM ${GLUT_INCLUDE_DIR})
#usage: target_link_libraries(${ProjNameApp} ${GLUT_LIBRARIES})
