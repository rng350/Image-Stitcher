cmake_minimum_required(VERSION 3.30)
project(Image_Stitcher)

set(CMAKE_CXX_STANDARD 20)
set(SOURCE_FILES main.cpp)

set(OpenCV_DIR "INSERT_OPENCV_MINGW_BUILD_PATH_HERE")
include_directories(${OpenCV_DIR}/include)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Image_Stitcher main.cpp)

target_link_libraries(Image_Stitcher
        ${OpenCV_LIBS}
        ${OpenCV_DIR}/lib/libopencv_core4110.dll.a
        ${OpenCV_DIR}/lib/libopencv_imgproc4110.dll.a
        ${OpenCV_DIR}/lib/libopencv_highgui4110.dll.a
        ${OpenCV_DIR}/lib/libopencv_imgcodecs4110.dll.a
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")