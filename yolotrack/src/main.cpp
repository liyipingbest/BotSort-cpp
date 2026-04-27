#include <iostream>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "YOLODetector.h"
#include "BoTSORT.h"
#include "DataType.h"
#include "track.h"

void draw_detections(cv::Mat& frame, const std::vector<yolo::Detection>& detections) {
    for (const auto& det : detections) {
        cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
        
        std::string label = "Class: " + std::to_string(det.class_id) + 
                         " Conf: " + std::to_string(det.confidence).substr(0, 4);
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 
                                           0.5, 1, &baseline);
        
        cv::Point text_origin(det.bbox.x, det.bbox.y - 5);
        if (text_origin.y < 0) text_origin.y = 0;
        
        cv::rectangle(frame, 
                   cv::Rect(text_origin.x, text_origin.y - text_size.height, 
                            text_size.width, text_size.height + baseline),
                   cv::Scalar(0, 255, 0), -1);
        
        cv::putText(frame, label, text_origin, cv::FONT_HERSHEY_SIMPLEX, 
                   0.5, cv::Scalar(0, 0, 0), 1);
    }
}

void draw_tracks(cv::Mat& frame, const std::vector<std::shared_ptr<Track>>& tracks) {
    static std::map<int, cv::Scalar> track_colors;
    
    for (const auto& track : tracks) {
        std::vector<float> tlwh = track->get_tlwh();
        cv::Rect bbox(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
        
        cv::Scalar color;
        if (track_colors.find(track->track_id) == track_colors.end()) {
            color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
            track_colors[track->track_id] = color;
        } else {
            color = track_colors[track->track_id];
        }
        
        cv::rectangle(frame, bbox, color, 2);
        
        std::string label = "ID: " + std::to_string(track->track_id);
        cv::putText(frame, label, cv::Point(bbox.x, bbox.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
    }
}

std::vector<Detection> convert_yolo_to_botsort(const std::vector<yolo::Detection>& yolo_detections) {
    std::vector<Detection> botsort_detections;
    
    for (const auto& yolo_det : yolo_detections) {
        Detection botsort_det;
        botsort_det.class_id = yolo_det.class_id;
        botsort_det.confidence = yolo_det.confidence;
        botsort_det.bbox_tlwh = cv::Rect_<float>(
            yolo_det.bbox.x, yolo_det.bbox.y,
            yolo_det.bbox.width, yolo_det.bbox.height
        );
        botsort_detections.push_back(botsort_det);
    }
    
    return botsort_detections;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <video_path> <yolo_model_path> [output_path]" << std::endl;
        std::cout << "  video_path: Path to input video file" << std::endl;
        std::cout << "  yolo_model_path: Path to YOLO model file (.onnx)" << std::endl;
        std::cout << "  output_path: (Optional) Path to save output video" << std::endl;
        return -1;
    }
    
    std::string video_path = argv[1];
    std::string yolo_model_path = argv[2];
    std::string output_path = (argc >= 4) ? argv[3] : "";
    
    std::cout << "Loading YOLO model from: " << yolo_model_path << std::endl;
    
    try {
        YOLODetector detector(yolo_model_path, 0.5f, 0.4f);
        
        std::cout << "Initializing BoTSORT tracker..." << std::endl;
        std::string tracker_config = "../config/tracker.ini";
        
        std::unique_ptr<BoTSORT> tracker;
        try {
            tracker = std::make_unique<BoTSORT>(tracker_config);
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
            std::cerr << "Using default tracker parameters..." << std::endl;
        }
        
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
            return -1;
        }
        
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        
        std::cout << "Video info: " << frame_width << "x" << frame_height 
                  << ", " << fps << " FPS, " << total_frames << " frames" << std::endl;
        
        cv::VideoWriter writer;
        if (!output_path.empty()) {
            int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');
            writer.open(output_path, fourcc, fps, cv::Size(frame_width, frame_height));
            if (!writer.isOpened()) {
                std::cerr << "Warning: Cannot create output video, will display only" << std::endl;
            }
        }
        
        cv::Mat frame;
        int frame_count = 0;
        double total_detect_time = 0.0;
        double total_track_time = 0.0;
        
        std::cout << "Starting processing..." << std::endl;
        
        while (cap.read(frame)) {
            auto start_detect = std::chrono::high_resolution_clock::now();
            std::vector<yolo::Detection> yolo_detections = detector.detect(frame);
            auto end_detect = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> detect_time = end_detect - start_detect;
            total_detect_time += detect_time.count();
            
            auto start_track = std::chrono::high_resolution_clock::now();
            std::vector<Detection> botsort_detections = convert_yolo_to_botsort(yolo_detections);
            std::vector<std::shared_ptr<Track>> tracks = tracker->track(botsort_detections, frame);
            auto end_track = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> track_time = end_track - start_track;
            total_track_time += track_time.count();
            
            draw_detections(frame, yolo_detections);
            draw_tracks(frame, tracks);
            
            cv::imshow("YOLO + BoTSORT Tracking", frame);
            
            if (writer.isOpened()) {
                writer.write(frame);
            }
            
            frame_count++;
            
            if (frame_count % 30 == 0) {
                std::cout << "Processed " << frame_count << "/" << total_frames 
                          << " frames | Detect: " << (30.0 / (detect_time.count() * 30)) 
                          << " FPS | Track: " << (30.0 / (track_time.count() * 30)) << " FPS" 
                          << std::endl;
            }
            
            if (cv::waitKey(1) == 27) {
                std::cout << "Processing stopped by user" << std::endl;
                break;
            }
        }
        
        cap.release();
        if (writer.isOpened()) {
            writer.release();
            std::cout << "Output video saved to: " << output_path << std::endl;
        }
        cv::destroyAllWindows();
        
        std::cout << "Processing completed!" << std::endl;
        std::cout << "Total frames processed: " << frame_count << std::endl;
        std::cout << "Average detection FPS: " << frame_count / total_detect_time << std::endl;
        std::cout << "Average tracking FPS: " << frame_count / total_track_time << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
