#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

namespace yolo {

struct Detection {
    int class_id;
    float confidence;
    cv::Rect bbox;
};

} // namespace yolo

class YOLODetector {
public:
    YOLODetector(const std::string& model_path, 
                float conf_threshold = 0.5f, 
                float nms_threshold = 0.4f);
    ~YOLODetector();

    std::vector<yolo::Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net_;
    float conf_threshold_;
    float nms_threshold_;
    std::vector<std::string> output_names_;
    
    void preprocess(const cv::Mat& frame, cv::Mat& blob);
    std::vector<yolo::Detection> postprocess(const std::vector<cv::Mat>& outputs, 
                                   const cv::Size& frame_size);
    cv::Mat resize_with_padding(const cv::Mat& frame, cv::Size& target_size);
};
