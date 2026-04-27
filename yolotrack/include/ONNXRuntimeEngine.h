#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "onnxruntime_c_api.h"


class ONNXRuntimeEngine {
public:
    ONNXRuntimeEngine(const std::string& model_path);
    ~ONNXRuntimeEngine();

    // Run inference on a preprocessed blob [1, 3, 640, 640] float32 NCHW
    std::vector<cv::Mat> forward(const cv::Mat& blob);

private:
    OrtEnv*        env_ = nullptr;
    OrtSession*    session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;

    std::vector<std::string>  input_names_;
    std::vector<std::string>  output_names_;
    std::vector<int64_t>      input_shape_;
};
