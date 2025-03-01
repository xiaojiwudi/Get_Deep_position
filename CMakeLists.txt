cmake_minimum_required(VERSION 3.10)
project(AprilTagDetector)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 使用 pkg-config 查找 apriltag 库
find_package(PkgConfig REQUIRED)
pkg_check_modules(APRILTAG REQUIRED apriltag)

# 包含头文件目录
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${APRILTAG_INCLUDE_DIRS}
)

# 添加可执行文件（假设代码文件为 apriltag_detector.cpp）
add_executable(apriltag_detector main.cpp)

# 链接 OpenCV 和 apriltag 库
target_link_libraries(apriltag_detector ${OpenCV_LIBS} ${APRILTAG_LIBRARIES})
