cmake_minimum_required(VERSION 3.5)

project(Neuro)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/eigen")
find_package (Eigen3 REQUIRED NO_MODULE)
add_definitions(-DDebug -glldb -g)

add_executable(Test Test_Field.cpp)
