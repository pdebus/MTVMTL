cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

find_package(OpenCV REQUIRED)
find_package(CGAL REQUIRED)
include(${CGAL_USE_FILE})

find_package(Cholmod REQUIRED)
find_package(SuperLU REQUIRED)

#find_package(MPI REQUIRED)
#find_package(hwloc REQUIRED)
#find_package(Scotch REQUIRED)
#find_package(Pastix REQUIRED)

#SET(CMAKE_C_COMPILER mpicc)
#SET(CMAKE_CXX_COMPILER mpicxx)

enable_testing()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/.. /usr/include/superlu /usr/include/eigen3 $ENV{HOME}/projects/iod)
add_definitions(-std=c++14 -g -fopenmp)
#add_definitions(-O3 -march=native)
add_definitions(-Ofast -march=native)
add_definitions(-DNDEBUG)

add_executable(spline_test spline_test.cpp)
add_test(spline_test spline_test )
target_link_libraries(spline_test ${CGAL_LIBRARIES} ${CGAL_3RD_PARTY_LIBRARIES})

add_executable(autodiff_test autodiff_test.cpp)
add_test(autodiff_test autodiff_test )
target_link_libraries(autodiff_test )

add_executable(mat_util_test mat_util_test.cpp)
add_test(mat_util_test mat_util_test )
target_link_libraries(mat_util_test )

add_executable(opengl_cube_test opengl_cube_test.cpp)
add_test(opengl_cube_test opengl_cube_test )
target_link_libraries(opengl_cube_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(opengl_image_cube_test opengl_image_cube_test.cpp)
add_test(opengl_image_cube_test opengl_image_cube_test )
target_link_libraries(opengl_image_cube_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES})

add_executable(opengl_ellipsoid_test opengl_ellipsoid_test.cpp)
add_test(opengl_ellipsoid_test opengl_ellipsoid_test )
target_link_libraries(opengl_ellipsoid_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(opengl_pyramid_test opengl_pyramid_test.cpp)
add_test(opengl_pyramid_test opengl_pyramid_test )
target_link_libraries(opengl_pyramid_test ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(manifold_son_test manifold_son_test.cpp)
add_test(manifold_son_test manifold_son_test)
target_link_libraries(manifold_son_test gomp)

add_executable(manifold_spd_test manifold_spd_test.cpp)
add_test(manifold_spd_test manifold_spd_test)
target_link_libraries(manifold_spd_test gomp)

add_executable(manifold_grassmann_test manifold_grassmann_test.cpp)
add_test(manifold_grassmann_test manifold_grassmann_test)
target_link_libraries(manifold_grassmann_test gomp)

add_executable(manifold_test manifold_test.cpp)
add_test(manifold_test manifold_test)

add_executable(data_test data_test.cpp)
add_test(data_test data_test)
target_link_libraries(data_test ${OpenCV_LIBS} gomp)

add_executable(data3d_test data3d_test.cpp)
add_test(data3d_test data3d_test)
target_link_libraries(data3d_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} gomp)

add_executable(sphere_data_test sphere_data_test.cpp)
add_test(sphere_data_test sphere_data_test)
target_link_libraries(sphere_data_test ${OpenCV_LIBS} gomp)

add_executable(son_data_test son_data_test.cpp)
add_test(son_data_test son_data_test )
target_link_libraries(son_data_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp)

add_executable(spd_data_test spd_data_test.cpp)
add_test(spd_data_test spd_data_test )
target_link_libraries(spd_data_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp)

add_executable(dti_data_test dti_data_test.cpp)
add_test(dti_data_test dti_data_test )
target_link_libraries(dti_data_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp)

add_executable(dti_data_test3d dti_data_test3d.cpp)
add_test(dti_data_test3d dti_data_test3d )
target_link_libraries(dti_data_test3d ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp)

add_executable(functional_test functional_test.cpp)
add_test(functional_test functional_test)
target_link_libraries(functional_test ${OpenCV_LIBS} gomp)

add_executable(functional3d_test functional3d_test.cpp)
add_test(functional3d_test functional3d_test)
target_link_libraries(functional3d_test ${OpenCV_LIBS} gomp)

add_executable(brightness_functional_test brightness_functional_test.cpp)
add_test(brightness_functional_test brightness_functional_test)
target_link_libraries(brightness_functional_test ${OpenCV_LIBS} gomp)

add_executable(chromaticity_functional_test chromaticity_functional_test.cpp)
add_test(chromaticity_functional_test chromaticity_functional_test)
target_link_libraries(chromaticity_functional_test ${OpenCV_LIBS} gomp)

add_executable(tvmin_test tvmin_test.cpp)
add_test(tvmin_test tvmin_test)
target_link_libraries(tvmin_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(tvmin_test_prpt tvmin_test_prpt.cpp)
add_test(tvmin_test_prpt tvmin_test_prpt)
target_link_libraries(tvmin_test_prpt ${OpenCV_LIBS} gomp)

add_executable(tvmin_test_prpt3d tvmin_test_prpt3d.cpp)
add_test(tvmin_test_prpt3d tvmin_test_prpt3d)
target_link_libraries(tvmin_test_prpt3d ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} gomp)

add_executable(inpaint_test inpaint_test.cpp)
add_test(inpaint_test inpaint_test)
target_link_libraries(inpaint_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(sphere_tvmin_test sphere_tvmin_test.cpp)
add_test(sphere_tvmin_test sphere_tvmin_test)
target_link_libraries(sphere_tvmin_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(sphere_tvmin_prpt_test sphere_tvmin_prpt_test.cpp)
add_test(sphere_tvmin_prpt_test sphere_tvmin_prpt_test)
target_link_libraries(sphere_tvmin_prpt_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(grassmann_chromaticnoise_test grassmann_chromaticnoise_test.cpp)
add_test(grassmann_chromaticnoise_test grassmann_chromaticnoise_test)
target_link_libraries(grassmann_chromaticnoise_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(son_functional_test son_functional_test.cpp)
add_test(son_functional_test son_functional_test)
target_link_libraries(son_functional_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(son_tvmin_test son_tvmin_test.cpp)
add_test(son_tvmin_test son_tvmin_test)
target_link_libraries(son_tvmin_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(spd_tvmin_test spd_tvmin_test.cpp)
add_test(spd_tvmin_test spd_tvmin_test)
target_link_libraries(spd_tvmin_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(dti_functional_test dti_functional_test.cpp)
add_test(dti_functional_test dti_functional_test)
target_link_libraries(dti_functional_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(dti_tvmin_test dti_tvmin_test.cpp)
add_test(dti_tvmin_test dti_tvmin_test)
target_link_libraries(dti_tvmin_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(dti_tvmin_prpt_test dti_tvmin_prpt_test.cpp)
add_test(dti_tvmin_prpt_test dti_tvmin_prpt_test)
target_link_libraries(dti_tvmin_prpt_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(dti_tvmin_prpt_test3d dti_tvmin_prpt_test3d.cpp)
add_test(dti_tvmin_prpt_test3d dti_tvmin_prpt_test3d)
target_link_libraries(dti_tvmin_prpt_test3d ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(son_inpainting_test son_inpainting_test.cpp)
add_test(son_inpainting_test son_inpainting_test)
target_link_libraries(son_inpainting_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(son_fingerprint_test son_fingerprint_test.cpp)
add_test(son_fingerprint_test son_fingerprint_test)
target_link_libraries(son_fingerprint_test ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(son_optical_flow son_optical_flow.cpp)
add_test(son_optical_flow son_optical_flow)
target_link_libraries(son_optical_flow ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} gomp ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

add_executable(colorization_test colorization_test.cpp)
add_test(colorization_test colorization_test)
target_link_libraries(colorization_test ${OpenCV_LIBS} gomp ${CHOLMOD_LIBRARIES} ${SUPERLU_LIBRARIES}) #${HWLOCK_LIBRARIES} ${SCOTCH_LIBRARIES} ${PASTIX_LIBRARIES})

