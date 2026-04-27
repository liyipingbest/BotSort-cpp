#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace yolo {

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bbox;
};

} // namespace yolo

class ONNXRuntimeEngine;

class YOLODetector {
public:
    YOLODetector(const std::string& model_path,
                float conf_threshold = 0.5f,
                float nms_threshold = 0.4f);
    ~YOLODetector();

    std::vector<yolo::Detection> detect(const cv::Mat& frame);

private:
    std::unique_ptr<ONNXRuntimeEngine> engine_;
    float conf_threshold_;
    float nms_threshold_;

    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    std::vector<yolo::Detection> postprocess(const std::vector<cv::Mat>& outputs,
                                   const cv::Size& frame_size);
    cv::Mat resize_with_padding(const cv::Mat& frame, cv::Size& target_size);
};
