---
sidebar_position: 5
title: "AI Model Deployment"
---

# AI Model Deployment

## Introduction to AI Model Deployment in Isaac ROS

AI model deployment in Isaac ROS involves the process of taking trained machine learning models and integrating them into real-time robotic applications. For humanoid robotics, this deployment must be optimized for low-latency inference on embedded hardware while maintaining the accuracy required for safe and effective robot operation. NVIDIA Isaac™ provides specialized tools and frameworks to streamline this process, leveraging GPU acceleration and TensorRT optimization.

## Isaac ROS AI Framework Overview

### 1. Isaac ROS DNN Inference Architecture

The Isaac ROS Deep Neural Network (DNN) inference framework provides GPU-accelerated model execution:

```cpp
// Isaac ROS DNN Inference Node
#include "isaac_ros_dnn_inference/dnn_inference_base_node.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

class IsaacAIPerceptionNode : public isaac_ros::dnn_inference::DnnInferenceBaseNode
{
public:
    explicit IsaacAIPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : DnnInferenceBaseNode("ai_perception_node", options)
    {
        // Initialize input and output topics
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "input_image", 10,
            std::bind(&IsaacAIPerceptionNode::ImageCallback, this, std::placeholders::_1));

        detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "detections", 10);

        // Initialize TensorRT engine
        InitializeTensorRTEngine();
    }

private:
    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Preprocess image for inference
        auto preprocessed_image = PreprocessImage(*msg);

        // Perform inference using TensorRT
        auto inference_result = PerformInference(preprocessed_image);

        // Postprocess results
        auto detections = PostprocessResults(inference_result, msg->header);

        // Publish detections
        detection_publisher_->publish(detections);
    }

    std::vector<uint8_t> PreprocessImage(const sensor_msgs::msg::Image& image_msg)
    {
        // Convert ROS image to format expected by TensorRT
        cv::Mat image = cv_bridge::toCvShare(image_msg, "bgr8")->image;

        // Resize to model input dimensions
        cv::resize(image, image, cv::Size(input_width_, input_height_));

        // Normalize pixel values (0-255 to 0-1 range)
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        // Convert to NCHW format (batch, channels, height, width)
        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        std::vector<uint8_t> input_tensor(input_size_);
        size_t channel_size = input_height_ * input_width_ * sizeof(float);

        for (int c = 0; c < 3; ++c) {
            memcpy(input_tensor.data() + c * channel_size,
                   channels[c].data,
                   channel_size);
        }

        return input_tensor;
    }

    InferenceResult PerformInference(const std::vector<uint8_t>& input_tensor)
    {
        InferenceResult result;

        // Copy input to GPU memory
        cudaMemcpy(input_buffer_, input_tensor.data(), input_tensor.size(),
                   cudaMemcpyHostToDevice);

        // Perform inference
        std::vector<void*> bindings = {input_buffer_, output_buffer_};
        bool status = context_->executeV2(bindings.data());

        if (!status) {
            RCLCPP_ERROR(this->get_logger(), "Inference execution failed");
            return result;
        }

        // Copy output from GPU memory
        cudaMemcpy(result.output_data.data(), output_buffer_,
                   output_size_, cudaMemcpyDeviceToHost);

        return result;
    }

    vision_msgs::msg::Detection2DArray PostprocessResults(
        const InferenceResult& result, const std_msgs::msg::Header& header)
    {
        vision_msgs::msg::Detection2DArray detections;
        detections.header = header;

        // Parse model outputs (assuming YOLO format)
        const float* output = reinterpret_cast<const float*>(result.output_data.data());

        for (int i = 0; i < max_detections_; ++i) {
            float confidence = output[i * output_stride_ + 4];

            if (confidence > confidence_threshold_) {
                vision_msgs::msg::Detection2D detection;
                detection.header = header;

                // Extract bounding box coordinates
                float x_center = output[i * output_stride_ + 0];
                float y_center = output[i * output_stride_ + 1];
                float width = output[i * output_stride_ + 2];
                float height = output[i * output_stride_ + 3];

                // Convert to image coordinates
                detection.bbox.center.x = x_center * input_width_;
                detection.bbox.center.y = y_center * input_height_;
                detection.bbox.size_x = width * input_width_;
                detection.bbox.size_y = height * input_height_;

                // Add confidence result
                vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
                hypothesis.hypothesis.class_id = GetMaxClassId(output + i * output_stride_ + 5);
                hypothesis.hypothesis.score = confidence;
                detection.results.push_back(hypothesis);

                detections.detections.push_back(detection);
            }
        }

        return detections;
    }

    void InitializeTensorRTEngine()
    {
        // Load TensorRT engine file
        std::string engine_path = this->declare_parameter("engine_file_path", "");
        if (engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Engine file path not specified");
            return;
        }

        // Load the serialized engine
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open engine file: %s", engine_path.c_str());
            return;
        }

        // Read engine data
        std::vector<char> engine_data;
        engine_file.seekg(0, engine_file.end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, engine_file.beg);
        engine_data.resize(size);
        engine_file.read(engine_data.data(), size);

        // Create runtime and engine
        runtime_ = nvinfer1::createInferRuntime(gLogger);
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size, nullptr);
        context_ = engine_->createExecutionContext();

        // Allocate GPU buffers
        AllocateGPUBuffers();

        RCLCPP_INFO(this->get_logger(), "TensorRT engine loaded successfully");
    }

    void AllocateGPUBuffers()
    {
        // Get binding indices
        input_index_ = engine_->getBindingIndex("input");
        output_index_ = engine_->getBindingIndex("output");

        // Get binding dimensions
        auto input_dims = engine_->getBindingDimensions(input_index_);
        auto output_dims = engine_->getBindingDimensions(output_index_);

        // Calculate buffer sizes
        input_size_ = 1; // Batch size
        for (int i = 0; i < input_dims.nbDims; ++i) {
            input_size_ *= input_dims.d[i];
        }
        input_size_ *= sizeof(float); // Assuming float32

        output_size_ = 1; // Batch size
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size_ *= output_dims.d[i];
        }
        output_size_ *= sizeof(float); // Assuming float32

        // Allocate GPU memory
        cudaMalloc(&input_buffer_, input_size_);
        cudaMalloc(&output_buffer_, output_size_);

        // Allocate output tensor
        output_tensor_size_ = output_size_ / sizeof(float);
    }

    struct InferenceResult {
        std::vector<uint8_t> output_data;
    };

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;

    // TensorRT components
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    void* input_buffer_{nullptr};
    void* output_buffer_{nullptr};

    // Model parameters
    int input_width_{640};
    int input_height_{640};
    int input_size_{0};
    int output_size_{0};
    size_t output_tensor_size_{0};
    int input_index_{-1};
    int output_index_{-1};
    float confidence_threshold_{0.5};
    int max_detections_{100};
    int output_stride_{85}; // YOLOv5 stride

    // Logger for TensorRT
    Logger gLogger;
};
```

### 2. Isaac ROS Triton Inference Server Integration

```cpp
// Isaac ROS integration with Triton Inference Server
#include "tritonclient/grpc/grpc_client.h"
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

class TritonAIPerceptionNode : public rclcpp::Node
{
public:
    TritonAIPerceptionNode() : Node("triton_ai_perception")
    {
        // Initialize Triton client
        InitializeTritonClient();

        // Create subscription and publisher
        image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "input_image", 10,
            std::bind(&TritonAIPerceptionNode::ImageCallback, this, std::placeholders::_1));

        detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "triton_detections", 10);
    }

private:
    void InitializeTritonClient()
    {
        // Get Triton server parameters
        std::string server_url = this->declare_parameter("triton_server_url", "localhost:8001");
        model_name_ = this->declare_parameter("model_name", "yolov5");
        model_version_ = this->declare_parameter("model_version", "1");

        // Create Triton client
        bool use_ssl = false;
        std::map<std::string, std::string> headers;
        long timeout = 0; // No timeout

        error_ = tc::InferenceServerGrpcClient::Create(&triton_client_, server_url, use_ssl, headers, timeout);

        if (!error_) {
            RCLCPP_INFO(this->get_logger(), "Connected to Triton server: %s", server_url.c_str());
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to connect to Triton server: %s", error_->Message().c_str());
        }
    }

    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!triton_client_ || error_) {
            return;
        }

        // Prepare inference request
        PrepareInferenceRequest(*msg);

        // Perform inference
        auto result = PerformTritonInference();

        // Process results
        if (result) {
            auto detections = ProcessInferenceResults(result, msg->header);
            detection_publisher_->publish(detections);
        }
    }

    void PrepareInferenceRequest(const sensor_msgs::msg::Image& image_msg)
    {
        // Preprocess image
        cv::Mat image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        cv::resize(image, image, cv::Size(640, 640));
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        // Prepare input data
        input_data_.resize(3 * 640 * 640); // CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(image, channels);

        size_t channel_size = 640 * 640;
        for (int c = 0; c < 3; ++c) {
            memcpy(input_data_.data() + c * channel_size,
                   channels[c].ptr<float>(),
                   channel_size * sizeof(float));
        }

        // Set up inputs
        inputs_.clear();
        tc::InferInput* input;
        error_ = tc::InferInput::Create(&input, "input", {1, 3, 640, 640}, "FP32");
        if (error_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create input: %s", error_->Message().c_str());
            return;
        }

        error_ = input->AppendRaw(reinterpret_cast<uint8_t*>(input_data_.data()),
                                 input_data_.size() * sizeof(float));
        if (error_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to append input data: %s", error_->Message().c_str());
            delete input;
            return;
        }

        inputs_.push_back(input);

        // Set up outputs
        outputs_.clear();
        tc::InferRequestedOutput* output;
        error_ = tc::InferRequestedOutput::Create(&output, "output");
        if (error_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create output: %s", error_->Message().c_str());
            delete input;
            return;
        }
        outputs_.push_back(output);
    }

    std::shared_ptr<tc::InferResult> PerformTritonInference()
    {
        std::shared_ptr<tc::InferResult> result;
        std::map<std::string, std::shared_ptr<tc::InferResult>> results;

        // Perform inference
        error_ = triton_client_->Infer(&results, model_name_, model_version_,
                                      inputs_, outputs_, headers_);

        if (!error_) {
            auto it = results.find("output");
            if (it != results.end()) {
                result = it->second;
            } else {
                RCLCPP_ERROR(this->get_logger(), "Output 'output' not found in results");
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Inference failed: %s", error_->Message().c_str());
        }

        // Clean up inputs and outputs
        for (auto& input : inputs_) {
            delete input;
        }
        for (auto& output : outputs_) {
            delete output;
        }

        return result;
    }

    vision_msgs::msg::Detection2DArray ProcessInferenceResults(
        const std::shared_ptr<tc::InferResult>& result,
        const std_msgs::msg::Header& header)
    {
        vision_msgs::msg::Detection2DArray detections;
        detections.header = header;

        // Get output data
        std::vector<uint8_t> output_data;
        error_ = result->RawData("output", &output_data);

        if (error_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get output data: %s", error_->Message().c_str());
            return detections;
        }

        // Parse detection results (assuming YOLO format)
        const float* output = reinterpret_cast<const float*>(output_data.data());
        size_t num_detections = output_data.size() / (sizeof(float) * 85); // YOLOv5 output format

        for (size_t i = 0; i < num_detections && i < 100; ++i) {
            float confidence = output[i * 85 + 4];
            if (confidence > 0.5) { // Confidence threshold
                vision_msgs::msg::Detection2D detection;
                detection.header = header;

                // Extract bounding box (x, y, width, height)
                detection.bbox.center.x = output[i * 85 + 0] * 640.0f; // Scale to image size
                detection.bbox.center.y = output[i * 85 + 1] * 640.0f;
                detection.bbox.size_x = output[i * 85 + 2] * 640.0f;
                detection.bbox.size_y = output[i * 85 + 3] * 640.0f;

                // Get class with highest probability
                int class_id = 0;
                float max_prob = 0.0f;
                for (int c = 5; c < 85; ++c) {
                    float prob = output[i * 85 + c];
                    if (prob > max_prob) {
                        max_prob = prob;
                        class_id = c - 5; // Adjust for YOLO format
                    }
                }

                vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
                hypothesis.hypothesis.class_id = std::to_string(class_id);
                hypothesis.hypothesis.score = confidence;
                detection.results.push_back(hypothesis);

                detections.detections.push_back(detection);
            }
        }

        return detections;
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;

    // Triton components
    std::unique_ptr<tc::InferenceServerGrpcClient> triton_client_;
    tc::Error error_;
    std::string model_name_;
    std::string model_version_;
    std::map<std::string, std::string> headers_;
    std::vector<tc::InferInput*> inputs_;
    std::vector<tc::InferRequestedOutput*> outputs_;
    std::vector<float> input_data_;
};
```

## Model Optimization with TensorRT

### 1. TensorRT Model Conversion and Optimization

```cpp
// TensorRT model optimization tools
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

class TensorRTOptimizer
{
public:
    struct ModelConfig {
        std::string onnx_model_path;
        std::string output_engine_path;
        int batch_size = 1;
        int input_height = 640;
        int input_width = 640;
        int input_channels = 3;
        std::vector<std::string> output_layer_names;
        bool use_fp16 = false;
        bool use_int8 = false;
        std::string calibration_dataset_path;
    };

    bool OptimizeModel(const ModelConfig& config)
    {
        // Create TensorRT builder
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to create TensorRT builder");
            return false;
        }

        // Create network definition
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = builder->createNetworkV2(explicitBatch);
        if (!network) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to create TensorRT network");
            return false;
        }

        // Create ONNX parser
        auto parser = nvonnxparser::createParser(*network, gLogger);
        if (!parser) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to create ONNX parser");
            return false;
        }

        // Parse ONNX model
        if (!parser->parseFromFile(config.onnx_model_path.c_str(),
                                  static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to parse ONNX model");
            return false;
        }

        // Configure builder
        auto profile = builder->createOptimizationProfile();
        auto input_tensor = network->getInput(0);

        // Set dynamic shape if needed
        if (input_tensor->isShapeTensor()) {
            nvinfer1::Dims input_dims{4, {config.batch_size, config.input_channels, config.input_height, config.input_width}};
            profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
            profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
            profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        }

        // Configure precision
        if (config.use_fp16) {
            builder->setFp16Mode(true);
        }
        if (config.use_int8) {
            builder->setInt8Mode(true);
            if (!SetupInt8Calibration(builder, network, config.calibration_dataset_path)) {
                RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to setup INT8 calibration");
                return false;
            }
        }

        // Set optimization profile
        builder->setConfigAttribute(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH, 1);
        builder->setMaxBatchSize(config.batch_size);

        // Build engine
        auto config_ptr = builder->createBuilderConfig();
        config_ptr->addOptimizationProfile(profile);

        auto engine = builder->buildEngineWithConfig(*network, *config_ptr);
        if (!engine) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to build TensorRT engine");
            return false;
        }

        // Serialize engine
        auto serialized_engine = engine->serialize();
        if (!serialized_engine) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to serialize TensorRT engine");
            return false;
        }

        // Save engine to file
        std::ofstream engine_file(config.output_engine_path, std::ios::binary);
        if (!engine_file) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"), "Failed to open output file: %s",
                        config.output_engine_path.c_str());
            return false;
        }

        engine_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        engine_file.close();

        RCLCPP_INFO(rclcpp::get_logger("tensorrt_optimizer"), "TensorRT engine saved to: %s",
                   config.output_engine_path.c_str());

        // Cleanup
        serialized_engine->destroy();
        engine->destroy();
        config_ptr->destroy();
        profile->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();

        return true;
    }

private:
    bool SetupInt8Calibration(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network,
                             const std::string& calibration_dataset_path)
    {
        // For INT8 calibration, we need to provide calibration data
        // This is a simplified example - in practice, you'd implement a custom calibrator
        if (calibration_dataset_path.empty()) {
            RCLCPP_ERROR(rclcpp::get_logger("tensorrt_optimizer"),
                        "Calibration dataset path required for INT8 optimization");
            return false;
        }

        // Create custom calibrator (simplified implementation)
        auto calibrator = std::make_unique<EntropyCalibrator2>(calibration_dataset_path);

        // This would typically involve:
        // 1. Loading calibration images
        // 2. Preprocessing them
        // 3. Providing them to TensorRT for calibration

        return true; // Simplified return
    }

    class EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
    {
    public:
        EntropyCalibrator2(const std::string& dataset_path) : dataset_path_(dataset_path) {}

        int getBatchSize() const noexcept override
        {
            return 1; // Simplified
        }

        bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override
        {
            // Load next batch of calibration data
            // This would involve loading images and preprocessing them
            return true; // Simplified
        }

        const void* readCalibrationCache(size_t& length) noexcept override
        {
            // Read calibration cache from file if it exists
            calibration_cache_.clear();
            std::ifstream cache_file(dataset_path_ + "/calibration.cache", std::ios::binary);
            if (cache_file) {
                cache_file.seekg(0, cache_file.end);
                size_t cache_size = cache_file.tellg();
                cache_file.seekg(0, cache_file.beg);

                calibration_cache_.resize(cache_size);
                cache_file.read(calibration_cache_.data(), cache_size);
                length = cache_size;
            } else {
                length = 0;
            }
            return length > 0 ? calibration_cache_.data() : nullptr;
        }

        void writeCalibrationCache(const void* cache, size_t length) noexcept override
        {
            // Write calibration cache to file
            std::ofstream cache_file(dataset_path_ + "/calibration.cache", std::ios::binary);
            if (cache_file) {
                cache_file.write(static_cast<const char*>(cache), length);
            }
        }

    private:
        std::string dataset_path_;
        std::vector<char> calibration_cache_;
    };

    Logger gLogger;
};
```

### 2. Isaac ROS Model Format Conversion

```cpp
// Isaac ROS model format conversion tools
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

class IsaacModelConverter
{
public:
    enum class ModelFormat {
        ONNX,
        TENSORRT,
        TORCHSCRIPT,
        TENSORFLOW,
        TFLITE
    };

    bool ConvertModel(const std::string& input_path, const std::string& output_path,
                     ModelFormat input_format, ModelFormat output_format)
    {
        switch (input_format) {
            case ModelFormat::ONNX:
                return ConvertFromONNX(input_path, output_path, output_format);
            case ModelFormat::TORCHSCRIPT:
                return ConvertFromTorchScript(input_path, output_path, output_format);
            case ModelFormat::TENSORFLOW:
                return ConvertFromTensorFlow(input_path, output_path, output_format);
            case ModelFormat::TFLITE:
                return ConvertFromTFLite(input_path, output_path, output_format);
            case ModelFormat::TENSORRT:
                RCLCPP_WARN(rclcpp::get_logger("model_converter"),
                           "TensorRT engine conversion is not supported as input format");
                return false;
        }

        return false;
    }

private:
    bool ConvertFromONNX(const std::string& input_path, const std::string& output_path,
                        ModelFormat output_format)
    {
        switch (output_format) {
            case ModelFormat::TENSORRT:
                return ConvertONNXToTensorRT(input_path, output_path);
            case ModelFormat::TORCHSCRIPT:
                return ConvertONNXToTorchScript(input_path, output_path);
            case ModelFormat::TFLITE:
                return ConvertONNXToTFLite(input_path, output_path);
            default:
                RCLCPP_ERROR(rclcpp::get_logger("model_converter"),
                           "Unsupported ONNX to %d conversion", static_cast<int>(output_format));
                return false;
        }
    }

    bool ConvertONNXToTensorRT(const std::string& onnx_path, const std::string& engine_path)
    {
        // Use TensorRT to convert ONNX to TensorRT engine
        TensorRTOptimizer optimizer;

        TensorRTOptimizer::ModelConfig config;
        config.onnx_model_path = onnx_path;
        config.output_engine_path = engine_path;
        config.use_fp16 = true; // Enable FP16 for better performance

        return optimizer.OptimizeModel(config);
    }

    bool ConvertONNXToTorchScript(const std::string& onnx_path, const std::string& torchscript_path)
    {
        // This would typically involve using ONNX-TorchScript conversion tools
        // or re-exporting from PyTorch
        RCLCPP_WARN(rclcpp::get_logger("model_converter"),
                   "ONNX to TorchScript conversion requires re-export from PyTorch");
        return false;
    }

    bool ConvertONNXToTFLite(const std::string& onnx_path, const std::string& tflite_path)
    {
        // Use ONNX-TFLite conversion (requires onnx-tf or similar tools)
        RCLCPP_WARN(rclcpp::get_logger("model_converter"),
                   "ONNX to TFLite conversion requires external tools like onnx-tf");
        return false;
    }

    bool ConvertFromTorchScript(const std::string& input_path, const std::string& output_path,
                               ModelFormat output_format)
    {
        // PyTorch-specific conversion
        RCLCPP_WARN(rclcpp::get_logger("model_converter"),
                   "TorchScript conversion requires PyTorch runtime");
        return false;
    }

    bool ConvertFromTensorFlow(const std::string& input_path, const std::string& output_path,
                              ModelFormat output_format)
    {
        // TensorFlow-specific conversion
        RCLCPP_WARN(rclcpp::get_logger("model_converter"),
                   "TensorFlow conversion requires TensorFlow runtime");
        return false;
    }

    bool ConvertFromTFLite(const std::string& input_path, const std::string& output_path,
                          ModelFormat output_format)
    {
        // TFLite-specific conversion
        RCLCPP_WARN(rclcpp::get_logger("model_converter"),
                   "TFLite conversion requires TensorFlow Lite runtime");
        return false;
    }
};
```

## Isaac ROS Tensor Management

### 1. Efficient Tensor Processing

```cpp
// Efficient tensor processing for Isaac ROS
#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"
#include "isaac_ros_managed_nh/managed_node_handle.hpp"

class EfficientTensorProcessor
{
public:
    EfficientTensorProcessor(rclcpp::Node* node) : node_(node)
    {
        // Initialize GPU memory pools
        InitializeMemoryPools();

        // Create tensor processing pipeline
        CreateProcessingPipeline();
    }

    TensorList ProcessTensors(const TensorList& input_tensors)
    {
        TensorList output_tensors;

        for (const auto& tensor : input_tensors.tensors) {
            // Determine processing method based on tensor properties
            if (tensor.data.size() > large_tensor_threshold_) {
                // Process large tensors asynchronously
                auto future_result = ProcessLargeTensorAsync(tensor);
                output_tensors.tensors.push_back(future_result.get());
            } else {
                // Process small tensors synchronously
                auto processed_tensor = ProcessSmallTensor(tensor);
                output_tensors.tensors.push_back(processed_tensor);
            }
        }

        return output_tensors;
    }

private:
    struct Tensor {
        std::string name;
        std::vector<int64_t> shape;
        std::string data_type; // FLOAT32, UINT8, etc.
        std::vector<uint8_t> data;
        std::map<std::string, std::string> metadata;
    };

    struct TensorList {
        std_msgs::msg::Header header;
        std::vector<Tensor> tensors;
    };

    void InitializeMemoryPools()
    {
        // Create GPU memory pools for efficient allocation
        cudaStreamCreate(&processing_stream_);
        cudaStreamCreate(&copy_stream_);

        // Pre-allocate common tensor sizes
        PreallocateCommonSizes();
    }

    void PreallocateCommonSizes()
    {
        // Pre-allocate memory for common tensor sizes to reduce allocation overhead
        std::vector<size_t> common_sizes = {
            640 * 640 * 3 * sizeof(float),  // Common image size
            224 * 224 * 3 * sizeof(float),  // ResNet input
            416 * 416 * 3 * sizeof(float),  // YOLO input
            1000 * sizeof(float)            // Common output size
        };

        for (size_t size : common_sizes) {
            void* buffer;
            cudaMalloc(&buffer, size);
            memory_pool_[size] = buffer;
        }
    }

    std::future<Tensor> ProcessLargeTensorAsync(const Tensor& input_tensor)
    {
        return std::async(std::launch::async, [this, input_tensor]() {
            return ProcessTensorOnGPU(input_tensor);
        });
    }

    Tensor ProcessSmallTensor(const Tensor& input_tensor)
    {
        // Process on CPU for small tensors (avoid GPU overhead)
        return ProcessTensorOnCPU(input_tensor);
    }

    Tensor ProcessTensorOnGPU(const Tensor& input_tensor)
    {
        Tensor output_tensor = input_tensor;

        // Allocate GPU memory
        void* d_input;
        void* d_output;

        size_t tensor_size = CalculateTensorSize(input_tensor);

        // Use memory pool if available, otherwise allocate
        auto pool_it = memory_pool_.find(tensor_size);
        if (pool_it != memory_pool_.end()) {
            d_input = pool_it->second;
        } else {
            cudaMalloc(&d_input, tensor_size);
        }

        cudaMalloc(&d_output, tensor_size);

        // Copy input to GPU
        cudaMemcpyAsync(d_input, input_tensor.data.data(), tensor_size,
                       cudaMemcpyHostToDevice, processing_stream_);

        // Process tensor (example: apply some transformation)
        ProcessTensorKernel<<<
            (tensor_size + 255) / 256, 256, 0, processing_stream_>>>(
            static_cast<float*>(d_input), static_cast<float*>(d_output),
            tensor_size / sizeof(float));

        // Copy result back to host
        output_tensor.data.resize(tensor_size);
        cudaMemcpyAsync(output_tensor.data.data(), d_output, tensor_size,
                       cudaMemcpyDeviceToHost, processing_stream_);

        // Synchronize stream
        cudaStreamSynchronize(processing_stream_);

        // Free GPU memory (or return to pool)
        cudaFree(d_output);

        return output_tensor;
    }

    Tensor ProcessTensorOnCPU(const Tensor& input_tensor)
    {
        Tensor output_tensor = input_tensor;

        // Simple CPU processing example
        if (input_tensor.data_type == "FLOAT32") {
            float* data = reinterpret_cast<float*>(output_tensor.data.data());
            size_t count = output_tensor.data.size() / sizeof(float);

            for (size_t i = 0; i < count; ++i) {
                data[i] = data[i] * 2.0f; // Example transformation
            }
        }

        return output_tensor;
    }

    size_t CalculateTensorSize(const Tensor& tensor)
    {
        size_t element_size = 4; // Default to float32

        if (tensor.data_type == "UINT8") element_size = 1;
        else if (tensor.data_type == "INT8") element_size = 1;
        else if (tensor.data_type == "FLOAT16") element_size = 2;
        else if (tensor.data_type == "FLOAT32") element_size = 4;
        else if (tensor.data_type == "FLOAT64") element_size = 8;

        size_t total_elements = 1;
        for (int64_t dim : tensor.shape) {
            total_elements *= dim;
        }

        return total_elements * element_size;
    }

    // CUDA kernel for tensor processing
    __global__ void ProcessTensorKernel(float* input, float* output, size_t size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // Example processing: normalize values
            float value = input[idx];
            output[idx] = fmaxf(0.0f, fminf(1.0f, value)); // Clamp to [0, 1]
        }
    }

    rclcpp::Node* node_;
    cudaStream_t processing_stream_;
    cudaStream_t copy_stream_;
    std::map<size_t, void*> memory_pool_;
    size_t large_tensor_threshold_ = 1024 * 1024; // 1MB threshold
};
```

## Model Deployment Strategies

### 1. Multi-Model Inference Pipeline

```cpp
// Multi-model inference pipeline for humanoid robotics
class MultiModelInferencePipeline
{
public:
    struct ModelInfo {
        std::string name;
        std::string engine_path;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<int> input_shapes;  // [batch, channels, height, width]
        std::vector<int> output_shapes;
        float confidence_threshold;
        int gpu_device;
    };

    MultiModelInferencePipeline(const std::vector<ModelInfo>& models)
    {
        // Initialize models on different GPU devices if available
        for (const auto& model_info : models) {
            auto model = std::make_unique<InferenceModel>(model_info);
            models_.push_back(std::move(model));
        }

        // Create processing threads for each model
        CreateProcessingThreads();
    }

    MultiModelResults ProcessMultiModel(const SensorData& sensor_data)
    {
        MultiModelResults results;

        // Distribute sensor data to appropriate models
        for (size_t i = 0; i < models_.size(); ++i) {
            auto& model = models_[i];

            // Select appropriate sensor data for this model
            auto model_input = PrepareModelInput(sensor_data, model->GetModelInfo());

            // Process asynchronously
            auto future_result = std::async(std::launch::async, [model, model_input]() {
                return model->Process(model_input);
            });

            results.model_results.push_back(future_result.get());
        }

        // Fuse results from all models
        results.fused_result = FuseModelResults(results.model_results);

        return results;
    }

private:
    struct SensorData {
        sensor_msgs::msg::Image::SharedPtr rgb_image;
        sensor_msgs::msg::Image::SharedPtr depth_image;
        sensor_msgs::msg::PointCloud2::SharedPtr pointcloud;
        sensor_msgs::msg::Imu::SharedPtr imu_data;
        geometry_msgs::msg::PoseStamped robot_pose;
    };

    struct MultiModelResults {
        std::vector<ModelResult> model_results;
        FusedResult fused_result;
    };

    struct ModelResult {
        std::string model_name;
        std::vector<Tensor> outputs;
        rclcpp::Time inference_time;
        float confidence;
    };

    struct FusedResult {
        std::vector<ObjectDetection> objects;
        std::vector<HumanPose> human_poses;
        std::vector<SemanticSegment> segmentation;
        geometry_msgs::msg::PoseStamped robot_state;
    };

    class InferenceModel
    {
    public:
        InferenceModel(const ModelInfo& info) : info_(info)
        {
            // Set GPU device
            cudaSetDevice(info.gpu_device);

            // Load TensorRT engine
            LoadTensorRTEngine(info.engine_path);
        }

        ModelResult Process(const std::vector<Tensor>& inputs)
        {
            ModelResult result;
            result.model_name = info_.name;

            // Copy inputs to GPU
            for (size_t i = 0; i < inputs.size(); ++i) {
                cudaMemcpy(input_buffers_[i], inputs[i].data.data(),
                          inputs[i].data.size(), cudaMemcpyHostToDevice);
            }

            // Perform inference
            auto start_time = std::chrono::high_resolution_clock::now();
            std::vector<void*> bindings = PrepareBindings();
            context_->executeV2(bindings.data());
            auto end_time = std::chrono::high_resolution_clock::now();

            result.inference_time = rclcpp::Time(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end_time - start_time).count());

            // Copy outputs from GPU
            for (size_t i = 0; i < output_buffers_.size(); ++i) {
                Tensor output;
                output.data.resize(output_sizes_[i]);
                cudaMemcpy(output.data.data(), output_buffers_[i],
                          output_sizes_[i], cudaMemcpyDeviceToHost);
                result.outputs.push_back(output);
            }

            return result;
        }

        const ModelInfo& GetModelInfo() const { return info_; }

    private:
        void LoadTensorRTEngine(const std::string& engine_path)
        {
            // Load serialized engine (implementation similar to previous examples)
            // Allocate input/output buffers
            // Initialize execution context
        }

        std::vector<void*> PrepareBindings()
        {
            std::vector<void*> bindings;
            for (auto buffer : input_buffers_) {
                bindings.push_back(buffer);
            }
            for (auto buffer : output_buffers_) {
                bindings.push_back(buffer);
            }
            return bindings;
        }

        ModelInfo info_;
        nvinfer1::IRuntime* runtime_{nullptr};
        nvinfer1::ICudaEngine* engine_{nullptr};
        nvinfer1::IExecutionContext* context_{nullptr};
        std::vector<void*> input_buffers_;
        std::vector<void*> output_buffers_;
        std::vector<size_t> input_sizes_;
        std::vector<size_t> output_sizes_;
    };

    std::vector<Tensor> PrepareModelInput(const SensorData& sensor_data, const ModelInfo& model_info)
    {
        std::vector<Tensor> inputs;

        // Prepare input tensors based on model requirements
        if (model_info.name.find("detection") != std::string::npos) {
            // Prepare for object detection model
            auto image_tensor = ConvertImageToTensor(sensor_data.rgb_image, model_info.input_shapes);
            inputs.push_back(image_tensor);
        }
        else if (model_info.name.find("pose") != std::string::npos) {
            // Prepare for pose estimation model
            auto pose_tensor = ConvertImageToTensor(sensor_data.rgb_image, model_info.input_shapes);
            inputs.push_back(pose_tensor);
        }
        else if (model_info.name.find("segmentation") != std::string::npos) {
            // Prepare for segmentation model
            auto seg_tensor = ConvertImageToTensor(sensor_data.rgb_image, model_info.input_shapes);
            inputs.push_back(seg_tensor);
        }

        return inputs;
    }

    Tensor ConvertImageToTensor(const sensor_msgs::msg::Image::SharedPtr& image_msg,
                               const std::vector<int>& target_shape)
    {
        Tensor tensor;
        tensor.shape = {target_shape[0], target_shape[1], target_shape[2], target_shape[3]};

        // Convert ROS image to tensor format
        cv::Mat image = cv_bridge::toCvShare(image_msg, "bgr8")->image;
        cv::resize(image, image, cv::Size(target_shape[3], target_shape[2]));

        // Normalize and convert to NCHW format
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(image, channels);

        tensor.data.resize(target_shape[1] * target_shape[2] * target_shape[3] * sizeof(float));
        size_t channel_size = target_shape[2] * target_shape[3] * sizeof(float);

        for (int c = 0; c < 3; ++c) {
            memcpy(tensor.data.data() + c * channel_size,
                   channels[c].data,
                   channel_size);
        }

        return tensor;
    }

    FusedResult FuseModelResults(const std::vector<ModelResult>& model_results)
    {
        FusedResult fused_result;

        for (const auto& result : model_results) {
            if (result.model_name.find("detection") != std::string::npos) {
                auto detections = ParseDetectionResults(result);
                fused_result.objects.insert(fused_result.objects.end(),
                                          detections.begin(), detections.end());
            }
            else if (result.model_name.find("pose") != std::string::npos) {
                auto poses = ParsePoseResults(result);
                fused_result.human_poses.insert(fused_result.human_poses.end(),
                                              poses.begin(), poses.end());
            }
            else if (result.model_name.find("segmentation") != std::string::npos) {
                auto segments = ParseSegmentationResults(result);
                fused_result.segmentation.insert(fused_result.segmentation.end(),
                                               segments.begin(), segments.end());
            }
        }

        return fused_result;
    }

    std::vector<ObjectDetection> ParseDetectionResults(const ModelResult& result)
    {
        std::vector<ObjectDetection> detections;
        // Parse detection model outputs
        // Implementation depends on model output format
        return detections;
    }

    std::vector<HumanPose> ParsePoseResults(const ModelResult& result)
    {
        std::vector<HumanPose> poses;
        // Parse pose estimation model outputs
        // Implementation depends on model output format
        return poses;
    }

    std::vector<SemanticSegment> ParseSegmentationResults(const ModelResult& result)
    {
        std::vector<SemanticSegment> segments;
        // Parse segmentation model outputs
        // Implementation depends on model output format
        return segments;
    }

    std::vector<std::unique_ptr<InferenceModel>> models_;
    std::vector<std::thread> processing_threads_;
};
```

### 2. Model Versioning and Management

```cpp
// Model versioning and management system
class ModelManager
{
public:
    struct ModelMetadata {
        std::string model_name;
        std::string version;
        std::string model_type; // detection, segmentation, pose_estimation, etc.
        std::string input_format; // RGB, RGBD, PointCloud, etc.
        std::vector<int> input_shape; // [batch, channels, height, width]
        std::vector<int> output_shape;
        float accuracy; // Model accuracy metric
        float inference_time; // Average inference time in ms
        std::string hardware_requirements; // GPU memory, compute capability
        rclcpp::Time created_time;
        std::string description;
        std::map<std::string, std::string> tags; // e.g., "indoor", "outdoor", "day", "night"
    };

    ModelManager(rclcpp::Node* node) : node_(node)
    {
        // Load model registry from file
        LoadModelRegistry();
    }

    bool RegisterModel(const std::string& model_path, const ModelMetadata& metadata)
    {
        // Validate model file exists and is accessible
        if (!std::filesystem::exists(model_path)) {
            RCLCPP_ERROR(node_->get_logger(), "Model file does not exist: %s", model_path.c_str());
            return false;
        }

        // Perform model validation
        if (!ValidateModel(model_path, metadata)) {
            RCLCPP_ERROR(node_->get_logger(), "Model validation failed for: %s", model_path.c_str());
            return false;
        }

        // Register model in the registry
        registered_models_[metadata.model_name + ":" + metadata.version] = {model_path, metadata};

        // Save updated registry
        SaveModelRegistry();

        RCLCPP_INFO(node_->get_logger(), "Model registered: %s version %s",
                   metadata.model_name.c_str(), metadata.version.c_str());

        return true;
    }

    std::string GetBestModel(const std::string& model_type,
                           const std::map<std::string, std::string>& requirements = {})
    {
        std::string best_model_path;
        float best_score = -1.0;

        for (const auto& [model_key, model_info] : registered_models_) {
            // Check if model type matches
            if (model_info.metadata.model_type != model_type) {
                continue;
            }

            // Check requirements
            bool meets_requirements = true;
            for (const auto& [req_key, req_value] : requirements) {
                if (req_key == "environment") {
                    auto it = model_info.metadata.tags.find(req_value);
                    if (it == model_info.metadata.tags.end()) {
                        meets_requirements = false;
                        break;
                    }
                }
                // Add other requirement checks as needed
            }

            if (!meets_requirements) {
                continue;
            }

            // Calculate score based on accuracy, inference time, and other factors
            float score = CalculateModelScore(model_info.metadata, requirements);

            if (score > best_score) {
                best_score = score;
                best_model_path = model_info.path;
            }
        }

        return best_model_path;
    }

    bool UpdateModel(const std::string& model_name, const std::string& new_model_path,
                    const ModelMetadata& new_metadata)
    {
        // Find existing model versions
        std::vector<std::string> existing_versions;
        for (const auto& [key, info] : registered_models_) {
            if (key.find(model_name + ":") == 0) {
                existing_versions.push_back(key.substr(model_name.length() + 1)); // Extract version
            }
        }

        // Validate new model
        if (!ValidateModel(new_model_path, new_metadata)) {
            RCLCPP_ERROR(node_->get_logger(), "New model validation failed");
            return false;
        }

        // Register new version
        std::string new_key = model_name + ":" + new_metadata.version;
        registered_models_[new_key] = {new_model_path, new_metadata};

        // Save updated registry
        SaveModelRegistry();

        RCLCPP_INFO(node_->get_logger(), "Model updated: %s to version %s",
                   model_name.c_str(), new_metadata.version.c_str());

        return true;
    }

    std::vector<ModelMetadata> GetModelHistory(const std::string& model_name)
    {
        std::vector<ModelMetadata> history;

        for (const auto& [key, info] : registered_models_) {
            if (key.find(model_name + ":") == 0) {
                history.push_back(info.metadata);
            }
        }

        // Sort by creation time
        std::sort(history.begin(), history.end(),
                 [](const ModelMetadata& a, const ModelMetadata& b) {
                     return a.created_time.nanoseconds() < b.created_time.nanoseconds();
                 });

        return history;
    }

private:
    struct RegisteredModel {
        std::string path;
        ModelMetadata metadata;
    };

    bool ValidateModel(const std::string& model_path, const ModelMetadata& metadata)
    {
        // Basic validation checks
        if (metadata.model_name.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "Model name is empty");
            return false;
        }

        if (metadata.version.empty()) {
            RCLCPP_ERROR(node_->get_logger(), "Model version is empty");
            return false;
        }

        // Check if model file has valid extension
        std::string extension = std::filesystem::path(model_path).extension();
        if (extension != ".engine" && extension != ".onnx" && extension != ".trt") {
            RCLCPP_ERROR(node_->get_logger(), "Invalid model file extension: %s", extension.c_str());
            return false;
        }

        // Additional validation can be added here
        // e.g., check if TensorRT engine is valid, etc.

        return true;
    }

    float CalculateModelScore(const ModelMetadata& metadata,
                            const std::map<std::string, std::string>& requirements)
    {
        float score = 0.0;

        // Accuracy contributes positively to score
        score += metadata.accuracy * 0.5;

        // Faster inference time contributes positively (inverse relationship)
        if (metadata.inference_time > 0) {
            score += (1.0 / metadata.inference_time) * 0.3;
        }

        // Check for specific requirements
        auto env_it = requirements.find("environment");
        if (env_it != requirements.end()) {
            auto tag_it = metadata.tags.find(env_it->second);
            if (tag_it != metadata.tags.end()) {
                score += 0.2; // Bonus for environment match
            }
        }

        return score;
    }

    void LoadModelRegistry()
    {
        std::string registry_path = node_->declare_parameter("model_registry_path",
                                                            "/tmp/model_registry.json");

        std::ifstream file(registry_path);
        if (!file.is_open()) {
            RCLCPP_WARN(node_->get_logger(), "Model registry file not found, starting fresh");
            return;
        }

        nlohmann::json registry_json;
        file >> registry_json;

        // Parse registry data
        if (registry_json.contains("models")) {
            for (const auto& model_json : registry_json["models"]) {
                RegisteredModel model_info;
                model_info.path = model_json["path"];

                ModelMetadata metadata;
                metadata.model_name = model_json["metadata"]["model_name"];
                metadata.version = model_json["metadata"]["version"];
                metadata.model_type = model_json["metadata"]["model_type"];
                metadata.accuracy = model_json["metadata"]["accuracy"];
                metadata.inference_time = model_json["metadata"]["inference_time"];

                model_info.metadata = metadata;

                std::string key = metadata.model_name + ":" + metadata.version;
                registered_models_[key] = model_info;
            }
        }

        file.close();
    }

    void SaveModelRegistry()
    {
        std::string registry_path = node_->declare_parameter("model_registry_path",
                                                            "/tmp/model_registry.json");

        nlohmann::json registry_json;
        nlohmann::json models_json = nlohmann::json::array();

        for (const auto& [key, model_info] : registered_models_) {
            nlohmann::json model_json;
            model_json["path"] = model_info.path;

            nlohmann::json metadata_json;
            metadata_json["model_name"] = model_info.metadata.model_name;
            metadata_json["version"] = model_info.metadata.version;
            metadata_json["model_type"] = model_info.metadata.model_type;
            metadata_json["accuracy"] = model_info.metadata.accuracy;
            metadata_json["inference_time"] = model_info.metadata.inference_time;

            model_json["metadata"] = metadata_json;
            models_json.push_back(model_json);
        }

        registry_json["models"] = models_json;

        std::ofstream file(registry_path);
        file << registry_json.dump(4);
        file.close();
    }

    rclcpp::Node* node_;
    std::map<std::string, RegisteredModel> registered_models_;
};
```

## Performance Monitoring and Optimization

### 1. Inference Performance Monitoring

```cpp
// Performance monitoring for AI inference
class InferencePerformanceMonitor
{
public:
    struct PerformanceMetrics {
        double average_inference_time_ms;
        double max_inference_time_ms;
        double min_inference_time_ms;
        double std_dev_inference_time_ms;
        int total_inferences;
        double throughput_fps; // Frames per second
        double gpu_utilization_percent;
        double memory_utilization_percent;
        int dropped_frames;
        double power_consumption_watts;
    };

    InferencePerformanceMonitor()
    {
        // Initialize monitoring thread
        monitoring_thread_ = std::thread(&InferencePerformanceMonitor::MonitoringLoop, this);
    }

    ~InferencePerformanceMonitor()
    {
        should_stop_ = true;
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
    }

    void StartInferenceTimer(const std::string& model_name)
    {
        inference_start_times_[model_name] = std::chrono::high_resolution_clock::now();
    }

    void EndInferenceTimer(const std::string& model_name)
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto start_time = inference_start_times_[model_name];
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        // Store inference time
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        inference_times_[model_name].push_back(duration.count() / 1000.0); // Convert to milliseconds

        // Limit stored samples to prevent memory overflow
        if (inference_times_[model_name].size() > max_samples_) {
            inference_times_[model_name].erase(inference_times_[model_name].begin());
        }
    }

    PerformanceMetrics GetMetrics(const std::string& model_name)
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);

        PerformanceMetrics metrics;
        const auto& times = inference_times_[model_name];

        if (times.empty()) {
            return metrics; // Return zero-initialized metrics
        }

        // Calculate statistics
        double sum = std::accumulate(times.begin(), times.end(), 0.0);
        metrics.average_inference_time_ms = sum / times.size();
        metrics.max_inference_time_ms = *std::max_element(times.begin(), times.end());
        metrics.min_inference_time_ms = *std::min_element(times.begin(), times.end());

        // Calculate standard deviation
        double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
        metrics.std_dev_inference_time_ms = std::sqrt(sq_sum / times.size() -
                                                     metrics.average_inference_time_ms * metrics.average_inference_time_ms);

        metrics.total_inferences = times.size();
        metrics.throughput_fps = 1000.0 / metrics.average_inference_time_ms; // Approximate FPS

        // Get GPU utilization (simplified)
        metrics.gpu_utilization_percent = GetGPUUtilization();
        metrics.memory_utilization_percent = GetGPUMemoryUtilization();

        return metrics;
    }

    void PublishMetrics(const std::string& model_name, rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub)
    {
        auto metrics = GetMetrics(model_name);

        std_msgs::msg::String msg;
        msg.data = "Model: " + model_name +
                  ", Avg Inference Time: " + std::to_string(metrics.average_inference_time_ms) + "ms" +
                  ", Throughput: " + std::to_string(metrics.throughput_fps) + " FPS" +
                  ", GPU Util: " + std::to_string(metrics.gpu_utilization_percent) + "%" +
                  ", Memory Util: " + std::to_string(metrics.memory_utilization_percent) + "%";

        pub->publish(msg);
    }

private:
    void MonitoringLoop()
    {
        while (!should_stop_) {
            // Collect system metrics periodically
            CollectSystemMetrics();

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    void CollectSystemMetrics()
    {
        // Collect GPU utilization, memory usage, etc.
        // This would typically involve calling nvidia-ml-py or similar
        current_gpu_util_ = GetSystemGPUUtilization();
        current_memory_util_ = GetSystemGPUMemoryUtilization();
    }

    double GetGPUUtilization()
    {
        // Return current GPU utilization for this model
        // In practice, this would be calculated based on the model's usage
        return current_gpu_util_;
    }

    double GetGPUMemoryUtilization()
    {
        // Return current GPU memory utilization for this model
        return current_memory_util_;
    }

    double GetSystemGPUUtilization()
    {
        // Simplified implementation - in practice, use nvidia-ml-py
        // or other system monitoring tools
        unsigned int utilization;
        // nvmlDeviceGetUtilizationRates(device, &utilization); // Example
        return 75.0; // Placeholder
    }

    double GetSystemGPUMemoryUtilization()
    {
        // Simplified implementation
        return 60.0; // Placeholder
    }

    std::map<std::string, std::chrono::high_resolution_clock::time_point> inference_start_times_;
    std::map<std::string, std::vector<double>> inference_times_; // In milliseconds
    std::mutex metrics_mutex_;

    std::thread monitoring_thread_;
    std::atomic<bool> should_stop_{false};

    double current_gpu_util_{0.0};
    double current_memory_util_{0.0};

    static constexpr size_t max_samples_ = 1000; // Maximum samples to store
};
```

## Next Steps

In the next section, we'll explore Isaac Sim integration for AI model training and validation, learning how to leverage Isaac Sim's realistic physics and rendering capabilities to generate synthetic training data and validate AI models before deployment on physical humanoid robots.