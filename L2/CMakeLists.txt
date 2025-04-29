cmake_minimum_required(VERSION 3.30)
project(L2)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)


add_executable(L2 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(L2 ${OpenCV_LIBS})
