# author: xieshufu@baidu.com

if ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
	set(WILDMAGIC5_ROOT /${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/wildmagic5/)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "14.04x86_64")
	set(WILDMAGIC5_ROOT /${CMAKE_SOURCE_DIR}/../../thirdparty/for_x86/wildmagic5/)
elseif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04aarch64")
	set(WILDMAGIC5_ROOT /${CMAKE_SOURCE_DIR}/../../thirdparty/for_arm/wildmagic5/)
endif ("$ENV{SYSTEM_ENVIRONMENT}" STREQUAL "16.04x86_64")
include_directories(SYSTEM ${WILDMAGIC5_ROOT}/include/)
link_directories(${WILDMAGIC5_ROOT}/lib/)
