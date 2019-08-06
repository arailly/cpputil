cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)

add_library(arailib arailib.cpp)
target_include_directories(arailib PRIVATE ${PROJECT_SOURCE_DIR}/include)