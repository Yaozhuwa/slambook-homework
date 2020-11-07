cmake_minimum_required(VERSION 3.2)

project(HomeWork7)

find_package(g2o REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${G2O_INCLUDE_DIRS})


add_executable(directba directBA.cpp)
target_link_libraries(directba ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
