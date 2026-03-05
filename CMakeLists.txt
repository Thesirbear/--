set(CMAKE_TOOLCHAIN_FILE
    "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")

cmake_minimum_required(VERSION 3.5)

project(neuro)

set(CMAKE_MAKE_PROGRAM "/usr/bin/make")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(folly  CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(glog   CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)

add_definitions(-DDebug -glldb -g)
add_executable(Test Test_Field.cpp)

target_link_libraries(Test PRIVATE
    Folly::folly
    Eigen3::Eigen
    gflags
    glog::glog
)

