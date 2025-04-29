cmake_minimum_required(VERSION 3.30)
project(project)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(Tesseract REQUIRED tesseract)


add_executable(project main.cpp
        PassportScanner.cpp)
include_directories(${OpenCV_INCLUDE_DIRS} ${Tesseract_INCLUDE_DIRS})
target_link_libraries(project ${OpenCV_LIBS} ${Tesseract_LIBRARIES})