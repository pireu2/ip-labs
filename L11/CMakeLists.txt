cmake_minimum_required(VERSION 3.30)
project(L11)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)


add_executable(L11 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(L11 ${OpenCV_LIBS})