# X64 to ARM64 cross-compile cmake config

# override the variable auto detectd by local_build.sh
set(ENV{SYSTEM_ENVIRONMENT} "16.04aarch64")
set(CROSS_COMPILE "16.04x86_64-16.04aarch64")

# build
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

# find path
set(CMAKE_FIND_ROOT_PATH $ENV{ARM64_ROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# deps
set(CMAKE_INCLUDE_PATH
	$ENV{ARM64_ROOT}/usr/include
	$ENV{ARM64_ROOT}/usr/include/aarch64-linux-gnu
	$ENV{ARM64_ROOT}/usr/local/driveworks/include
)
set(CMAKE_LIBRARY_PATH
	$ENV{ARM64_ROOT}/lib
	$ENV{ARM64_ROOT}/lib/aarch64-linux-gnu
	$ENV{ARM64_ROOT}/usr/lib
	$ENV{ARM64_ROOT}/usr/lib/aarch64-linux-gnu
	$ENV{ARM64_ROOT}/usr/local/lib
	$ENV{ARM64_ROOT}/usr/local/driveworks/lib
)
set(LD_LIBRARY_PATH
	$ENV{ARM64_ROOT}/lib
	$ENV{ARM64_ROOT}/lib/aarch64-linux-gnu
	$ENV{ARM64_ROOT}/usr/lib
	$ENV{ARM64_ROOT}/usr/lib/aarch64-linux-gnu
	$ENV{ARM64_ROOT}/usr/local/lib
	$ENV{ARM64_ROOT}/usr/local/driveworks/lib
)

#set(CMAKE_VERBOSE_MAKEFILE ON)
