

set(CMAKE_TOOLCHAIN_FILE
"${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
CACHE STRING "Vcpkg toolchain file")

cmake_minimum_required(VERSION 3.9)

project(neuro)
add_library(Neuro_lib INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(folly  CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(glog   CONFIG REQUIRED)
find_package(eigen3 CONFIG REQUIRED)

if (APPLE) 
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX) 
# Без OpenMp программа будет считать матрицы в однопотоке,
# что может быть очень медленно

#Чтобы clang-format работал
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Чтобы автодопы на редакторе работали
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_CURRENT_SOURCE_DIR}/build/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
endif()

find_program(CLANG_FORMAT clang-format)

file(GLOB_RECURSE Neuro_source CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuroLib/Block/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuroLib/Network/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuroLib/Dataloader/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuroLib/Scheduler/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuroLib/Optimizer/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/NeuroLib/LossFunction/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/cifar_10_test/*.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(Neuro_lib INTERFACE
    Folly::folly
    Eigen3::Eigen
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
    OpenMP::OpenMP_CXX
    # "-framework Accelerate" # Добавить если есть Apple Accelerate
)

target_compile_options(Cifar10_test PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=native -funroll-loops>
)

# add_executable(TestGrad tests/TestBlock.cpp)

# target_include_directories(TestGrad PRIVATE
#     EigenRand
# )

# target_link_libraries(TestGrad PRIVATE
#     Neuro_lib
#     Folly::folly
#     Eigen3::Eigen
#     gflags
#     glog::glog
#     OpenMP::OpenMP_CXX
# )

