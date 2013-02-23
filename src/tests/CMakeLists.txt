cmake_minimum_required(VERSION 2.6)
project(math-tests C)

add_executable(test_snlmath_sse2 test_simd.c)
add_executable(test_snlmath_sse3 test_simd.c)
target_link_libraries(test_snlmath_sse2 snlmath-sse2)
target_link_libraries(test_snlmath_sse3 snlmath-sse3)
set_target_properties(test_snlmath_sse2 PROPERTIES COMPILE_FLAGS -msse2)
set_target_properties(test_snlmath_sse3 PROPERTIES COMPILE_FLAGS -msse3)

add_test(snlmath_sse2 test_snlmath_sse2)
add_test(snlmath_sse3 test_snlmath_sse3)

