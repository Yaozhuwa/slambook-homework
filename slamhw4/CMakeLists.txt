cmake_minimum_required(VERSION 3.0)

project(hw4)
find_package(OpenCV 3.0 REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
add_definitions(-std=c++11)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(gaussnewton gaussnewton.cpp)
target_link_libraries(gaussnewton ${OpenCV_LIBS})

add_executable(undistort undistort_image.cpp)
target_link_libraries(undistort ${OpenCV_LIBS})

add_executable(disparity disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES})