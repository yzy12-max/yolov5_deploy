
# # ONNX
# # requirement of cmake version
# cmake_minimum_required(VERSION 3.5)
# # project name
# PROJECT(onnx CXX)

# # set the directory of executable files
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)  # 输出文件夹路径
# set(CMAKE_BUILD_TYPE "Release")  # 编译类型


# # find required opencv
# find_package(OpenCV REQUIRED)   # 使用find_package获取opencv的相关信息,如lib、include、version等
# # directory of opencv headers
# include_directories(${OpenCV_INCLUDE_DIRS}) # 头文件夹

# #onnxruntime
# set(ONNXRUNTIME_ROOT_PATH /data1/yzycode/opencv/onnxruntime-linux-x64-1.6.0/) # 根目录
# set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/)   # 头文件夹
# set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}lib/libonnxruntime.so)  # 依赖库

# # name of executable file and path of source file
# add_executable(onnx src/ort.cpp)
# # directory of opencv library
# link_directories(${OpenCV_LIBRARY_DIRS})    # 指定第三方库所在路径,该路径添加到第三方库搜索路径中
# # opencv libraries
# target_link_libraries(onnx ${OpenCV_LIBS})  # 相对路径

# include_directories(${ONNXRUNTIME_INCLUDE_DIRS})    # 头文件
# target_link_libraries(onnx ${ONNXRUNTIME_LIB})  # 依赖


# TRT 
# requirement of cmake version
cmake_minimum_required(VERSION 3.5)
# project name
PROJECT(yolov5_trt CXX)
set(CMAKE_CXX_STANDARD 17)
# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)  # 输出文件夹路径
set(CMAKE_BUILD_TYPE "Release")  # 编译类型


# find required opencv
find_package(OpenCV REQUIRED)   # 使用find_package获取opencv的相关信息,如lib、include、version等
find_package(glog REQUIRED)

# tensorrt
set(TENSORRT_ROOT_PATH /opt/TensorRT-8.0.1.6/) # 根目录
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT_PATH}/include/)   # 头文件夹
set(TENSORRT_LIB ${TENSORRT_ROOT_PATH}lib/libnvinfer.so ${TENSORRT_ROOT_PATH}lib/libnvonnxparser.so)  # 依赖动态库


# cuda
set(CUDA_ROOT_PATH /usr/local/cuda-10.2/) # 根目录
set(CUDA_INCLUDE_DIRS ${CUDA_ROOT_PATH}/include/)   # 头文件夹
set(CUDA_LIB ${CUDA_ROOT_PATH}lib64/libcudart.so)  # 依赖动态库

# name of executable file and path of source file
add_executable(yolov5_trt src/yolov5_trt.cpp)

# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}) # 头文件夹
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})    # 指定第三方库所在路径,该路径添加到第三方库搜索路径中
# opencv libraries
target_link_libraries(yolov5_trt ${OpenCV_LIBS} ${TENSORRT_LIB} ${CUDA_LIB} ${glog_LIB})  # 相对路径
