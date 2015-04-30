cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. /usr/include/eigen3 $ENV{HOME}/projects/iod)
add_definitions(-std=c++14 -g -fopenmp)
add_definitions(-Ofast -march=native)
add_definitions(-DNDEBUG)

add_executable(manifold_test manifold_test.cpp)
add_test(manifold_test manifold_test)

add_executable(data_test data_test.cpp)
add_test(data_test data_test)
target_link_libraries(data_test ${OpenCV_LIBS} gomp)

add_executable(functional_test functional_test.cpp)
add_test(functional_test functional_test)
target_link_libraries(functional_test ${OpenCV_LIBS} gomp)


