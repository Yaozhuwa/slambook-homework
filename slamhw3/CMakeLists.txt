cmake_minimum_required(VERSION 3.5)

project(slamhw3)
find_package(Pangolin 0.4 REQUIRED)
add_definitions(-std=c++11)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(q6 q6.cpp)
add_executable(q7 q7.cpp)
target_link_libraries(q6 ${Pangolin_LIBRARIES})
target_link_libraries(q7 ${Pangolin_LIBRARIES})