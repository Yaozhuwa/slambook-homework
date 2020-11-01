cmake_minimum_required(VERSION 3.2)

project(HomeWork5)

find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(orb computeORB.cpp)
target_link_libraries(orb ${OpenCV_LIBS})

add_executable(e2rt E2Rt.cpp)

add_executable(ba GN-BA.cpp)

add_executable(icp icp.cpp)
target_link_libraries(icp ${Pangolin_LIBRARIES})
