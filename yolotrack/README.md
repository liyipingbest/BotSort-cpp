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

## 画框识别参数与坐标映射

涉及检测框位置和数量的关键参数和代码位置：

### 1. YOLO 检测置信度与 NMS（`src/main.cpp:92`）

```cpp
YOLODetector detector(yolo_model_path, 0.5f, 0.4f);
//                                       ^^^^  ^^^^
//                                 conf_thresh  nms_thresh
```

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `conf_threshold` | 0.5 | 只有置信度高于此值的检测框才会进入 NMS。降低此值（如 0.25）可检测到更远/更小的目标 |
| `nms_threshold` | 0.4 | NMS IoU 阈值。两个检测框 IoU 超过此值则保留置信度高的，丢弃低的 |

### 2. 坐标映射：Letterbox 填充还原（`src/YOLODetector.cpp`）

YOLO 模型输入为 640×640，视频帧通过 letterbox 缩放并居中后送入模型。**检测坐标必须正确还原到原始帧**，核心公式在 `postprocess()` 中：

```cpp
// resize_with_padding() 中保存的 letterbox 参数：
letterbox_scale_  = min(640/frame_width, 640/frame_height);  // 缩放比例
letterbox_x_offset_ = (640 - new_width)  / 2;                 // 水平填充偏移
letterbox_y_offset_ = (640 - new_height) / 2;                 // 垂直填充偏移

// postprocess() 中还原原始帧坐标：
x = (cx - w/2 - letterbox_x_offset_) / letterbox_scale_;
y = (cy - h/2 - letterbox_y_offset_) / letterbox_scale_;
width  = w / letterbox_scale_;
height = h / letterbox_scale_;
```

对于 16:9 视频（如 1920×1080），`y_offset ≈ 140`，图像内容在 640×640 空间中只占 y=[140, 500]。如果直接用 `y = cy × (frame_height/640)` 会错误地将检测框压缩到画面中部。

### 3. BoTSORT 跟踪器配置（`../config/tracker.ini`）

```
track_high_thresh = 0.4     # 高于此值的检测用于第一级关联匹配
track_low_thresh  = 0.1     # 低于此值的检测直接丢弃
new_track_thresh  = 0.7     # 高于此值的检测才能启动新的跟踪轨迹
track_buffer      = 30      # 轨迹丢失后保留的帧数（实际 = frame_rate/30 × track_buffer）
match_thresh      = 0.7     # 第一级关联的最大匹配代价
proximity_thresh  = 0.5     # IoU 距离阈值（1-IoU），超过则拒绝匹配
```

**画框少的调参建议**：
- `new_track_thresh` 降到 0.4，让更多检测能新建轨迹
- `track_high_thresh` 和 `track_low_thresh` 默认合理，一般不需调整
- 优先降低 YOLO 的 `conf_threshold`（影响预 NMS 候选数量）

### 4. 检测→跟踪→渲染 流程（`src/main.cpp`）

```
cap.read(frame)
    │
    ▼
detector.detect(frame)          ← YOLO 推理 + letterbox 坐标还原 + NMS
    │
    ▼
convert_yolo_to_botsort()       ← 转为 BoTSORT 的 Detection 格式
    │
    ▼
tracker->track(detections, frame) ← Kalman 预测 + 两级关联匹配
    │
    ▼
draw_detections(frame, ...)     ← 绿色框 = YOLO 原始检测
draw_tracks(frame, ...)         ← 彩色框 = BoTSORT 跟踪轨迹
    │
    ▼
cv::imshow(...)                 ← 显示 / writer.write() 保存
```

- `draw_detections()` (`src/main.cpp:12`): 画绿色检测框
- `draw_tracks()` (`src/main.cpp:36`): 画彩色跟踪框（同 track_id 同颜色）
- BoTSORT 内部的 clamp (`botsort/src/BoTSORT.cpp:161-168`): 将检测框裁剪到画面边界内

### 5. 诊断检测框分布的快速方法

在 `YOLODetector::postprocess()` 中临时加入以下 debug 代码，按 y 坐标分段统计 pre-NMS 检测数量：

```cpp
int y_bins[6] = {0};
for (int i = 0; i < num_detections; ++i) {
    // ... 计算 y 坐标 ...
    int bin = y * 6 / frame_size.height;
    y_bins[std::max(0, std::min(5, bin))]++;
}
```

如果某分段始终为 0，说明模型在该区域没有检测到目标（不是代码问题）。

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
