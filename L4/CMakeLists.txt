cmake_minimum_required(VERSION 3.30)
project(L4)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)


add_executable(L4 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(L4 ${OpenCV_LIBS})
