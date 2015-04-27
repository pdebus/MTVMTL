cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. /usr/include/eigen3 $ENV{HOME}/projects/iod)
add_definitions(-std=c++14)

add_executable(manifold_test manifold_test.cpp)
add_test(manifold_test manifold_test)



