cmake_minimum_required(VERSION 2.8.3)
project(cudasift_ros)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-DVERBOSE") 
  endif()
endif()

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES cudasift_ros
  CATKIN_DEPENDS cv_bridge image_transport roscpp
  DEPENDS eigen opencv
)

set(cuda_sources
  # dynamic.cu
  src/cudaImage.cu  
  src/cudaImage.h  
  src/cudaSiftH.cu 
  src/cudaSiftH.h  
  src/matching.cu  
  src/cudaSiftD.h  
  src/cudaSift.h  
  src/cudautils.h
)  

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

SET(CUDA_SEPARABLE_COMPILATION ON)

 
install(FILES 
  ${cuda_sources} 
  ${sources}
  cudaSiftD.cu
  CMakeLists.txt
  Copyright.txt
  DESTINATION .
)

cuda_add_executable(cudasift_ros_node src/cudaSift_ros.cpp ${cuda_sources} OPTIONS -arch=sm_35)
target_link_libraries(cudasift_ros_node
  ${catkin_LIBRARIES} ${OpenCV_LIBS} /usr/local/cuda/lib64/libcudadevrt.a
)

