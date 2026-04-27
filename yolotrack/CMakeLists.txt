CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# Set project name
project(yolotrack VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add conda lib path and onnxruntime path to rpath
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
    set(ORT_LIB_DIR "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/onnxruntime/capi")
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib;${ORT_LIB_DIR}")
    set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib;${ORT_LIB_DIR}")
endif()

# Set Build Type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find OpenCV (prefer conda env over system)
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/cmake/opencv4")
    set(OpenCV_DIR "$ENV{CONDA_PREFIX}/lib/cmake/opencv4")
    set(OpenCV_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
else()
    set(OpenCV_DIR /usr/lib/x86_64-linux-gnu/cmake/opencv4)
endif()
find_package(OpenCV REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Find CUDA (optional for YOLO inference)
find_package(CUDA QUIET)

# Find BoTSORT
set(BOTSORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../)
if(EXISTS ${BOTSORT_DIR}/botsort)
    add_subdirectory(${BOTSORT_DIR}/botsort ${CMAKE_BINARY_DIR}/botsort)
    set(BOTSORT_INCLUDE_DIRS ${BOTSORT_DIR}/botsort/include)
    set(BOTSORT_LIBRARIES botsort)
else()
    message(WARNING "BoTSORT directory not found, please set correct path")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${BOTSORT_INCLUDE_DIRS}
)

# Find ONNX Runtime
if(DEFINED ENV{CONDA_PREFIX})
    set(ONNXRUNTIME_LIB_DIR "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/onnxruntime/capi")
else()
    set(ONNXRUNTIME_LIB_DIR "/usr/lib")
endif()
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR} REQUIRED)

# Source files
set(SOURCES
    src/main.cpp
    src/YOLODetector.cpp
    src/ONNXRuntimeEngine.cpp
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${BOTSORT_LIBRARIES}
    ${ONNXRUNTIME_LIB}
)

# Add CUDA support if available
if(CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# Add NO_CUDA macro if CUDA is not found
if(NOT CUDA_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE NO_CUDA)
endif()

# Install rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
