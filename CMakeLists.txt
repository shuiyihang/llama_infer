cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(llama_infer CXX)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug or Release)" FORCE)

#----------------------------------------------------
# 第三方库

# git submodule add git@github.com:nlohmann/json.git third_party/json
add_subdirectory(third_party/json)

# git submodule add https://gitlab.com/conradsnicta/armadillo-code.git third_party/armadillo-code
add_subdirectory(third_party/armadillo-code)


# 添加 Abseil
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)  # 禁用 Abseil 安装
add_subdirectory(third_party/abseil-cpp)

# 添加 RE2
# 修改re2中顶层cmake 注释禁用 RE2 安装
add_subdirectory(third_party/re2)

#-----------------------------------------------------


include_directories(${CMAKE_SOURCE_DIR}/inc)


# 将目录项下所有源文件添加到变量
aux_source_directory(${CMAKE_SOURCE_DIR}/src DIR_SRC)
add_library(llama SHARED ${DIR_SRC})
target_link_libraries(llama PRIVATE openblas)

# PRIVATE仅当前目标需要用
# PUBLIC 当前目标需要用，依赖它的其他目标也会继承这些头文件
target_include_directories(llama PRIVATE ${CMAKE_SOURCE_DIR}/third_party/json/include)
target_include_directories(llama PRIVATE ${CMAKE_SOURCE_DIR}/third_party/armadillo-code/include)
target_include_directories(llama PRIVATE ${CMAKE_SOURCE_DIR}/third_party/re2)
target_include_directories(llama PRIVATE ${CMAKE_SOURCE_DIR}/third_party/abseil-cpp)


# add_subdirectory(test)
add_subdirectory(main)

set(FORMAT_DIR "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/inc" "${CMAKE_SOURCE_DIR}/main")
add_custom_target(
    format
    COMMAND find ${FORMAT_DIR} -name '*.h' -o -name '*.cc' | xargs clang-format -i -style=file
)