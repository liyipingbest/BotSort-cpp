#include "ONNXRuntimeEngine.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

#define ORT_ABORT_ON_ERROR(expr) do {                               \
    OrtStatus* s = (expr);                                          \
    if (s != nullptr) {                                             \
        const char* msg = ort_api->GetErrorMessage(s);              \
        std::string err(msg);                                       \
        ort_api->ReleaseStatus(s);                                  \
        throw std::runtime_error("ONNX Runtime error: " + err);     \
    }                                                               \
} while(0)


static inline const OrtApi* get_ort_api() {
    return OrtGetApiBase()->GetApi(ORT_API_VERSION);
}


ONNXRuntimeEngine::ONNXRuntimeEngine(const std::string& model_path) {
    const OrtApi* ort_api = get_ort_api();

    ORT_ABORT_ON_ERROR(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "yolotrack", &env_));

    OrtSessionOptions* session_opts = nullptr;
    ORT_ABORT_ON_ERROR(ort_api->CreateSessionOptions(&session_opts));
    ORT_ABORT_ON_ERROR(ort_api->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL));
    ORT_ABORT_ON_ERROR(ort_api->SetIntraOpNumThreads(session_opts, 4));
    ORT_ABORT_ON_ERROR(ort_api->SetInterOpNumThreads(session_opts, 1));

    ORT_ABORT_ON_ERROR(ort_api->CreateSession(env_, model_path.c_str(), session_opts, &session_));
    ort_api->ReleaseSessionOptions(session_opts);

    ORT_ABORT_ON_ERROR(ort_api->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault,
                                                  &memory_info_));

    // YOLO models have fixed IO names
    input_names_  = {"images"};
    output_names_ = {"output0"};
    input_shape_  = {1, 3, 640, 640};

    std::cout << "ONNX Runtime engine loaded: " << model_path << std::endl;
}


ONNXRuntimeEngine::~ONNXRuntimeEngine() {
    const OrtApi* ort_api = get_ort_api();
    if (memory_info_)  ort_api->ReleaseMemoryInfo(memory_info_);
    if (session_)      ort_api->ReleaseSession(session_);
    if (env_)          ort_api->ReleaseEnv(env_);
}


std::vector<cv::Mat> ONNXRuntimeEngine::forward(const cv::Mat& blob) {
    const OrtApi* ort_api = get_ort_api();
    const float* data = reinterpret_cast<const float*>(blob.data);
    size_t total = blob.total();

    std::vector<int64_t> in_shape = input_shape_;

    OrtValue* input_tensor = nullptr;
    ORT_ABORT_ON_ERROR(ort_api->CreateTensorWithDataAsOrtValue(
        memory_info_, (void*)data, total * sizeof(float),
        in_shape.data(), in_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor));

    std::vector<const char*> in_names_c;
    for (auto& n : input_names_) in_names_c.push_back(n.c_str());

    std::vector<const char*> out_names_c;
    for (auto& n : output_names_) out_names_c.push_back(n.c_str());

    std::vector<OrtValue*> output_tensors(output_names_.size(), nullptr);

    OrtRunOptions* run_opts = nullptr;
    ORT_ABORT_ON_ERROR(ort_api->CreateRunOptions(&run_opts));

    ORT_ABORT_ON_ERROR(ort_api->Run(
        session_, run_opts,
        in_names_c.data(), (const OrtValue* const*)&input_tensor, 1,
        out_names_c.data(), out_names_c.size(),
        output_tensors.data()));

    ort_api->ReleaseRunOptions(run_opts);
    ort_api->ReleaseValue(input_tensor);

    std::vector<cv::Mat> outputs;
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        OrtTensorTypeAndShapeInfo* shape_info = nullptr;
        ORT_ABORT_ON_ERROR(ort_api->GetTensorTypeAndShape(output_tensors[i], &shape_info));

        size_t num_dims = 0;
        ORT_ABORT_ON_ERROR(ort_api->GetDimensionsCount(shape_info, &num_dims));
        std::vector<int64_t> dims64(num_dims);
        ORT_ABORT_ON_ERROR(ort_api->GetDimensions(shape_info, dims64.data(), num_dims));
        ort_api->ReleaseTensorTypeAndShapeInfo(shape_info);

        void* out_data = nullptr;
        ORT_ABORT_ON_ERROR(ort_api->GetTensorMutableData(output_tensors[i], &out_data));

        std::vector<int> dims(dims64.begin(), dims64.end());
        cv::Mat mat(static_cast<int>(num_dims), dims.data(), CV_32F, out_data);
        outputs.push_back(mat.clone());
        ort_api->ReleaseValue(output_tensors[i]);
    }

    return outputs;
}
