cmake_minimum_required(VERSION 3.0)
project( SharpnessAssess )

# Use C++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
message(STATUS "C++11 support has been enabled by default.")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")

#headfile
include_directories(include)

#library
link_directories(build)
add_library(seeta_fa_lib SHARED src/cfan.cpp src/face_alignment.cpp src/sift.cpp)
set(required_libs seeta_fa_lib)

find_package( OpenCV REQUIRED )
list(APPEND required_libs ${OpenCV_LIBS} seeta_facedet_lib)

#executable
add_executable( Sharpness main.cpp )
target_link_libraries( Sharpness  ${required_libs} )
