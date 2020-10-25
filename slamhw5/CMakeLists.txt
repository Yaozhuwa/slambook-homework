cmake_minimum_required(VERSION 3.2)

project(HomeWork5)

find_package(OpenCV REQUIRED)

add_executable(orb computeORB.cpp)
target_link_libraries(orb ${OpenCV_LIBS})

add_executable(e2rt E2Rt.cpp)

add_executable(ba GN-BA.cpp)