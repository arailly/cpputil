cmake_minimum_required(VERSION 3.14)

add_executable(test arailib-test.cpp)

target_link_libraries(test gtest gtest_main)
target_link_libraries(test arailib)
target_include_directories(test PRIVATE ${PROJECT_SOURCE_DIR}/include)

# omp option
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")