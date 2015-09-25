cmake_minimum_required (VERSION 2.8)

project (MTVMTL)

#include_directories(/usr/include/eigen3)

#add_definitions(-std=c++14)
add_subdirectory(thirdparty/vpp)
add_subdirectory(thirdparty/iod)
#add_subdirectory(test)

install(DIRECTORY mtvmtl DESTINATION include FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY thirdparty/SparseSuiteSupport DESTINATION include/mtvmtl FILES_MATCHING PATTERN "*.cmake")
