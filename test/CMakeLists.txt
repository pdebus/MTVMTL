cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

find_package(Cholmod REQUIRED)

#find_package(MPI REQUIRED)
#find_package(hwloc REQUIRED)
#find_package(Scotch REQUIRED)
#find_package(Pastix REQUIRED)

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. /usr/include/eigen3 $ENV{HOME}/projects/iod)
add_definitions(-std=c++14 -g -fopenmp)
add_definitions(-Ofast -march=native)
add_definitions(-DNDEBUG)

add_executable(spline_test spline_test.cpp)
add_test(spline_test spline_test )
target_link_libraries(spline_test ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})

add_executable(manifold_test manifold_test.cpp)
add_test(manifold_test manifold_test)

add_executable(data_test data_test.cpp)
add_test(data_test data_test)
target_link_libraries(data_test ${OpenCV_LIBS} gomp)

add_executable(functional_test functional_test.cpp)
add_test(functional_test functional_test)
target_link_libraries(functional_test ${OpenCV_LIBS} gomp)

add_executable(tvmin_test tvmin_test.cpp)
add_test(tvmin_test tvmin_test)
target_link_libraries(tvmin_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

