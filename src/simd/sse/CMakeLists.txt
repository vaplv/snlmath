cmake_minimum_required(VERSION 2.6)
project(snlmath-sse C)

file(GLOB SSE_FILES_INC *.h)
file(GLOB SSE_FILES ${SSE_FILES_INC} ${SIMD_FILES_INC} *.c ../regular/*.c)

add_library(snlmath-sse2 SHARED ${SSE_FILES})
add_library(snlmath-sse3 SHARED ${SSE_FILES})
set_target_properties(snlmath-sse2 PROPERTIES DEFINE_SYMBOL SIMD_SHARED_BUILD)
set_target_properties(snlmath-sse3 PROPERTIES DEFINE_SYMBOL SIMD_SHARED_BUILD)
set_target_properties(snlmath-sse2 PROPERTIES COMPILE_FLAGS -msse2)
set_target_properties(snlmath-sse3 PROPERTIES COMPILE_FLAGS -msse3)

install(TARGETS snlmath-sse2 snlmath-sse3 LIBRARY DESTINATION lib)
install(FILES ${SSE_FILES_INC} DESTINATION include/snlmath/simd/sse)
