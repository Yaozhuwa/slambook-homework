cmake_minimum_required(VERSION 3.2)

project(HomeWork7)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/home/yyz/SDK/g2o/cmake_modules")
find_package(G2O REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# find g2o lib
find_package(G2O REQUIRED)
IF(G2O_FOUND)
    include_directories(${G2O_INCLUDE_DIR})
    message("G2O lib is found:"${G2O_INCLUDE_DIR})
ENDIF(G2O_FOUND)

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )

add_executable(directba directBA.cpp)
# target_link_libraries(directba ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})
target_link_libraries(directba ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS})

add_executable(dba DBA.cpp)
target_link_libraries(dba ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${G2O_LIBS})