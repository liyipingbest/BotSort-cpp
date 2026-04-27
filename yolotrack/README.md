# YOLOTrack - YOLOv11 + BoTSORT 目标检测与跟踪系统

这个项目集成了 YOLOv11 目标检测算法和 BoTSORT 多目标跟踪算法，用于视频中的实时目标检测与跟踪。

## 功能特性

- **YOLOv11 检测**: 使用 OpenCV DNN 模块进行高效的目标检测
- **BoTSORT 跟踪**: 集成 BoTSORT 算法进行多目标跟踪
- **实时处理**: 支持实时视频处理和可视化
- **结果保存**: 可选择保存处理后的视频

## 项目结构

```
yolotrack/
├── include/           # 头文件
│   └── YOLODetector.h
├── src/              # 源文件
│   ├── main.cpp
│   └── YOLODetector.cpp
├── config/           # 配置文件
├── models/           # 模型文件
├── output/           # 输出结果
└── CMakeLists.txt
```

## 依赖项

- OpenCV 4.x (包含 DNN 模块)
- CMake 3.16+
- C++17 编译器
- BoTSORT-cpp 项目

## 编译方法

```bash
cd yolotrack
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## 使用方法

### 基本用法

```bash
./bin/yolotrack <video_path> <yolo_model_path> [output_path]
```

### 参数说明

- `video_path`: 输入视频文件路径
- `yolo_model_path`: YOLO 模型文件路径 (.onnx 格式)
- `output_path`: (可选) 输出视频文件路径

### 示例

```bash
./bin/yolotrack input.mp4 models/yolov11n.onnx output.mp4
```

## YOLO 模型要求

- 模型格式: ONNX
- 输入尺寸: 640x640
- 输出格式: [num_detections, 4 + num_classes]
  - 前4个值: [cx, cy, width, height]
  - 后续值: 各类别的置信度

## BoTSORT 配置

BoTSORT 跟踪器使用 `../BoTSORT-cpp/config/tracker.ini` 配置文件。如需自定义配置，请修改该文件。

## 键盘控制

- `ESC`: 退出程序

## 性能优化

- 支持多线程处理
- 可配置的置信度阈值和 NMS 阈值
- 实时 FPS 显示

## 注意事项

1. 确保 YOLO 模型文件路径正确
2. BoTSORT-cpp 项目必须在同级目录
3. 视频格式支持: MP4, AVI, MKV 等
4. 建议使用 GPU 加速 YOLO 推理（需要 CUDA）

## 故障排除

### 编译错误

如果遇到编译错误，请检查：
- OpenCV 是否正确安装
- BoTSORT-cpp 路径是否正确
- CMake 版本是否符合要求

### 运行时错误

如果遇到运行时错误，请检查：
- YOLO 模型文件是否存在
- 视频文件是否可读
- BoTSORT 配置文件是否存在

## 许可证

本项目基于 BoTSORT-cpp 项目，遵循相应的开源许可证。
