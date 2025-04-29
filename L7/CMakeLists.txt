cmake_minimum_required(VERSION 3.30)
project(L7)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)


add_executable(L7 main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(L7 ${OpenCV_LIBS})
