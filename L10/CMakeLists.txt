cmake_minimum_required(VERSION 3.30)
project(L10)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)


add_executable(L10 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(L10 ${OpenCV_LIBS})