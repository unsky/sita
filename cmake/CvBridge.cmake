set(CvBridge_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/cv-bridge/include)
set(CvBridge_LIBRARIY_DIRS ${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/cv-bridge/lib)
include_directories(CvBridge_INCLUDE_DIRS)
link_directories(CvBridge_LIBRARIY_DIRS)
file(GLOB CvBridge_LIBS ${CvBridge_LIBRARIY_DIRS}/*.so)
