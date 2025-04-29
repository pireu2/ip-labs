cmake_minimum_required(VERSION 3.30)
project(L6)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)


add_executable(L6 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(L6 ${OpenCV_LIBS})
