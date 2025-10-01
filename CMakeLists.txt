cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Debug)
project(arima CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

include_directories(include lib)

file(GLOB_RECURSE SOURCES src/*.cpp)

set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)
set(OBJ_DIR ${CMAKE_SOURCE_DIR}/build)

add_executable(arima ${SOURCES})

set_target_properties(arima PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
)

target_compile_definitions(arima PRIVATE VERBOSE DEBUG)
