

set(CMAKE_TOOLCHAIN_FILE
"${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
CACHE STRING "Vcpkg toolchain file")

cmake_minimum_required(VERSION 3.5)

project(neuro)
add_library(Neuro_lib INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(folly  CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(glog   CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)

#Чтобы clang-format работал
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Чтобы автодопы на редакторе работали
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/compile_commands.json)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/build/compile_commands.json
    ${CMAKE_SOURCE_DIR}/compile_commands.json)
endif()

find_program(CLANG_FORMAT clang-format)

file(GLOB_RECURSE Neuro_source CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/include/NeuroLib/Block/*.hpp
    ${CMAKE_SOURCE_DIR}/include/NeuroLib/Network/*.hpp
    ${CMAKE_SOURCE_DIR}/include/NeuroLib/Dataset/*.hpp
    ${CMAKE_SOURCE_DIR}/include/NeuroLib/Scheduler/*.hpp
    ${CMAKE_SOURCE_DIR}/include/NeuroLib/Optimizer/*.hpp
    ${CMAKE_SOURCE_DIR}/include/NeuroLib/LossFunction/*.hpp
    ${CMAKE_SOURCE_DIR}/tests/cifar_10_test/*.cpp
)

if (CLANG_FORMAT)
    add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${Neuro_source}
    COMMENT "clang-format"
    )
endif()


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#Сборка библиотеки
target_include_directories(Neuro_lib INTERFACE
${CMAKE_SOURCE_DIR}/include
)

add_executable(Cifar10_test tests/cifar_10_test/Mlp_cifar_10.cpp)

target_include_directories(Cifar10_test PRIVATE
    EigenRand
)

target_link_libraries(Cifar10_test PRIVATE
    Neuro_lib
    Folly::folly
    Eigen3::Eigen
    gflags
    glog::glog
)

target_compile_options(Cifar10_test PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-matrch=native -funroll-loops>
)



