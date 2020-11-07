cmake_minimum_required(VERSION 3.2)

project(HomeWork6)
find_package(OpenCV REQUIRED)

add_executable(oflow optical_flow.cpp)
target_link_libraries(oflow ${OpenCV_LIBS})


add_executable(direct direct_method.cpp)
target_link_libraries(direct ${OpenCV_LIBS})


add_executable(disparity oflow_disparity.cpp)
target_link_libraries(disparity ${OpenCV_LIBS})