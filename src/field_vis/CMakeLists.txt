cmake_minimum_required(VERSION 3.0.2)
project(field_vis)

set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fPIC")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    visualization_msgs
    sensor_msgs
)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

add_executable(field_vis src/field_vis.cpp)
target_link_libraries(field_vis ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(lbfgs_test src/lbfgs.cpp)
target_link_libraries(lbfgs_test ${catkin_LIBRARIES} ${PCL_LIBRARIES})