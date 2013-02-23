cmake_minimum_required(VERSION 2.6)
project(snlmath C)
enable_testing()

################################################################################
# Setup compiler
################################################################################
if(NOT CMAKE_COMPILER_IS_GNUCC)
  message(FATAL_ERROR "Unsupported compiler")
endif(NOT CMAKE_COMPILER_IS_GNUCC)

set(CMAKE_DEBUG_POSTFIX "-dbg")

set(CMAKE_C_FLAGS "-pedantic -std=c99 -Wall -Wextra -Wcast-align -Wmissing-declarations -Wmissing-prototypes -fvisibility=hidden -fstrict-aliasing -Wl,-z,defs")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math")

################################################################################
# Check dependencies
################################################################################
find_path(SNLSYS_INCLUDE_DIR "snlsys/snlsys.h")
if(NOT SNLSYS_INCLUDE_DIR)
  message(FATAL_ERROR "Can't find snlsys headers")
endif()

include_directories(${SNLSYS_INCLUDE_DIR})

################################################################################
# Sub projects
################################################################################
add_subdirectory(simd)
add_subdirectory(tests)

