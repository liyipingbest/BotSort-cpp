#include "YOLODetector.h"
#include "ONNXRuntimeEngine.h"
#include <algorithm>

YOLODetector::YOLODetector(const std::string& model_path,
                           float conf_threshold,
                           float nms_threshold)
    : conf_threshold_(conf_threshold), nms_threshold_(nms_threshold) {
    engine_ = std::make_unique<ONNXRuntimeEngine>(model_path);
}

YOLODetector::~YOLODetector() {}

cv::Mat YOLODetector::resize_with_padding(const cv::Mat& frame, cv::Size& target_size) {
    int original_width = frame.cols;
    int original_height = frame.rows;

    letterbox_scale_ = std::min(static_cast<float>(target_size.width) / original_width,
                         static_cast<float>(target_size.height) / original_height);

    int new_width = static_cast<int>(original_width * letterbox_scale_);
    int new_height = static_cast<int>(original_height * letterbox_scale_);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_width, new_height));

    cv::Mat padded(target_size, CV_8UC3, cv::Scalar(114, 114, 114));

    letterbox_x_offset_ = (target_size.width - new_width) / 2;
    letterbox_y_offset_ = (target_size.height - new_height) / 2;

    cv::Rect roi(letterbox_x_offset_, letterbox_y_offset_, new_width, new_height);
    resized.copyTo(padded(roi));

    return padded;
}

void YOLODetector::preprocess(const cv::Mat& frame, cv::Mat& blob) {
    cv::Size input_size(640, 640);
    cv::Mat padded = resize_with_padding(frame, input_size);

    cv::dnn::blobFromImage(padded, blob, 1.0/255.0, input_size, cv::Scalar(0, 0, 0), true, false);
}

std::vector<yolo::Detection> YOLODetector::postprocess(const std::vector<cv::Mat>& outputs,
                                            const cv::Size& frame_size) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_scale = 1.0f / letterbox_scale_;
    float y_scale = 1.0f / letterbox_scale_;

    for (const auto& output : outputs) {
        const float* data = (float*)output.data;
        int num_classes = output.size[1] - 4;
        int num_detections = output.size[2];

        for (int i = 0; i < num_detections; ++i) {
            float cx = data[0 * num_detections + i];
            float cy = data[1 * num_detections + i];
            float w  = data[2 * num_detections + i];
            float h  = data[3 * num_detections + i];

            float max_conf = 0.0f;
            int max_class_id = 0;

            for (int j = 0; j < num_classes; ++j) {
                float conf = data[(4 + j) * num_detections + i];
                if (conf > max_conf) {
                    max_conf = conf;
                    max_class_id = j;
                }
            }

            if (max_conf > conf_threshold_) {

                int x = static_cast<int>((cx - w/2.0f - letterbox_x_offset_) * x_scale);
                int y = static_cast<int>((cy - h/2.0f - letterbox_y_offset_) * y_scale);
                int width = static_cast<int>(w * x_scale);
                int height = static_cast<int>(h * y_scale);

                x = std::max(0, std::min(x, frame_size.width - 1));
                y = std::max(0, std::min(y, frame_size.height - 1));
                width = std::min(width, frame_size.width - x);
                height = std::min(height, frame_size.height - y);

                boxes.push_back(cv::Rect(x, y, width, height));
                confidences.push_back(max_conf);
                class_ids.push_back(max_class_id);
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    std::vector<yolo::Detection> detections;
    for (int idx : indices) {
        yolo::Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.bbox = boxes[idx];
        detections.push_back(det);
    }

    return detections;
}

std::vector<yolo::Detection> YOLODetector::detect(const cv::Mat& frame) {
    cv::Mat blob;
    preprocess(frame, blob);

    std::vector<cv::Mat> outputs = engine_->forward(blob);

    return postprocess(outputs, frame.size());
}
