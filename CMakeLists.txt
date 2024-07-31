cmake_minimum_required(VERSION 3.5)

project(YOLOv8Det LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(MY_HOME "C:/Users/bobod/Documents/YOLOv8Det/")#This is the only code that needs to be changed.


#Eigen
set(EIGEN3_INCLUDE_DIR "${MY_HOME}/eigen")
include_directories(${EIGEN3_INCLUDE_DIR})


# OpenCV
set(OpenCV_DIR "${MY_HOME}/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



#yolov8
set(YOLOv8_INCLUDE_DIR "${MY_HOME}/yolov8")
include_directories(${YOLOv8_INCLUDE_DIR})


include(GNUInstallDirs)




add_executable(YOLOv8DetFunction main_function.cpp
    ${YOLOv8_INCLUDE_DIR}/inference.cpp)


add_executable(YOLOv8DetClasses main_classes.cpp
    ${YOLOv8_INCLUDE_DIR}/inference.cpp)



add_executable(YOLOv8DetOOP main_OOP.cpp
    ${YOLOv8_INCLUDE_DIR}/inference.cpp
)


# 链接OpenCV库
target_link_libraries(YOLOv8DetFunction ${OpenCV_LIBS} )
target_link_libraries(YOLOv8DetClasses ${OpenCV_LIBS} )
target_link_libraries(YOLOv8DetOOP ${OpenCV_LIBS} )

