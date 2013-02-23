cmake_minimum_required(VERSION 2.6)
project(snlmath-simd C)

file(GLOB SIMD_FILES_INC *.h)
install(FILES ${SIMD_FILES_INC} DESTINATION include/snlmath/simd/)

add_subdirectory(sse)

