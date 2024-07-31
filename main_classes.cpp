// Author: shaoshengsong
#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "inference.h"
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>

using namespace std;
using namespace cv;

// 线程安全队列类定义
template <typename T>
class ThreadSafeQueue {
public:
    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
        condition_.notify_one();
    }

    bool tryPop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = queue_.front();
        queue_.pop();
        return true;
    }

    bool waitAndPop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        value = queue_.front();
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
};

// 帧和推理结果的结构体
struct FrameResult {
    cv::Mat frame;
    std::vector<Detection> detections;
};

// 读取视频帧的类
class VideoReader {
public:
    VideoReader(const std::string& videoFilePath, ThreadSafeQueue<cv::Mat>& frameQueue, int framesPerSecond)
        : videoFilePath_(videoFilePath), frameQueue_(frameQueue), framesPerSecond_(framesPerSecond) {}

    void operator()() {
        cv::VideoCapture cap(videoFilePath_);

        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int frameInterval = static_cast<int>(fps / framesPerSecond_);

        cv::Mat frame;
        int frameCount = 0;

        while (true) {
            cap >> frame;
            if (frame.empty()) {
                break; // 视频结束
            }

            if (frameCount % frameInterval == 0) {
                frameQueue_.push(frame.clone());
                std::cout << "Frame " << frameCount << " added to queue." << std::endl;
            }

            frameCount++;
        }

        cap.release();
        std::cout << "Video reading completed. Total frames added to queue: " << frameCount / frameInterval << std::endl;
    }

private:
    std::string videoFilePath_;
    ThreadSafeQueue<cv::Mat>& frameQueue_;
    int framesPerSecond_;
};

// 处理帧并推理的类
class FrameProcessor {
public:
    FrameProcessor(ThreadSafeQueue<cv::Mat>& frameQueue, ThreadSafeQueue<FrameResult>& resultQueue, Inference& inf)
        : frameQueue_(frameQueue), resultQueue_(resultQueue), inf_(inf) {}

    void operator()() {
        while (true) {
            cv::Mat frame;
            if (frameQueue_.waitAndPop(frame)) {
                std::vector<Detection> output = inf_.runInference(frame);
                FrameResult frameResult = { frame, output };
                resultQueue_.push(frameResult);
            } else {
                break; // 没有更多帧可处理
            }
        }
    }

private:
    ThreadSafeQueue<cv::Mat>& frameQueue_;
    ThreadSafeQueue<FrameResult>& resultQueue_;
    Inference& inf_;
};

// 保存推理结果到视频文件的类
class ResultSaver {
public:
    ResultSaver(ThreadSafeQueue<FrameResult>& resultQueue, const std::string& outputFilePath, int fps, cv::Size frameSize)
        : resultQueue_(resultQueue), outputFilePath_(outputFilePath), fps_(fps), frameSize_(frameSize) {}

    void operator()() {
        cv::VideoWriter writer(outputFilePath_, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps_, frameSize_);

        if (!writer.isOpened()) {
            std::cerr << "Error: Could not open the output video file for writing." << std::endl;
            return;
        }

        while (true) {
            FrameResult frameResult;
            if (resultQueue_.waitAndPop(frameResult)) {
                cv::Mat frame = frameResult.frame;
                std::vector<Detection> output = frameResult.detections;

                int detections = output.size();
                std::cout << "Number of detections:" << detections << std::endl;

                for (int i = 0; i < detections; ++i) {
                    Detection detection = output[i];

                    cv::Rect box = detection.box;
                    cv::Scalar color = detection.color;

                    cv::rectangle(frame, box, color, 2);

                    std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
                    cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                    cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                    cv::rectangle(frame, textBox, color, cv::FILLED);
                    cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
                }

                writer.write(frame);
            } else {
                break; // 没有更多帧可处理
            }
        }

        writer.release();
        std::cout << "Video writing completed." << std::endl;
    }

private:
    ThreadSafeQueue<FrameResult>& resultQueue_;
    std::string outputFilePath_;
    int fps_;
    cv::Size frameSize_;
};

int main(int argc, char** argv) {
    // 默认视频文件路径为 1.MP4
    namespace fs = std::filesystem;
    fs::path current_path = fs::current_path();
    std::string videoFilePath = (argc == 2) ? argv[1] : current_path.string() + "/1.mp4";
    std::cout << "videoFilePath " << videoFilePath << std::endl;
    int framesPerSecond = 1; // 每秒钟保存1帧

    ThreadSafeQueue<cv::Mat> frameQueue;
    ThreadSafeQueue<FrameResult> resultQueue;

    // 创建并启动读取视频帧的线程
    VideoReader videoReader(videoFilePath, frameQueue, framesPerSecond);
    std::thread videoThread(videoReader);

    // 获取视频帧尺寸和帧率
    cv::VideoCapture cap(videoFilePath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));
    cv::Size frameSize(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))
        );
    cap.release();

    std::string projectBasePath = current_path.string() + "/ultralytics";
    bool runOnGPU = false;

    Inference inf(projectBasePath + "/yolov8s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    // 创建并启动处理帧的线程，将帧的推理结果保存到另一个队列
    FrameProcessor frameProcessor(frameQueue, resultQueue, inf);
    std::thread processThread(frameProcessor);

    // 创建并启动保存推理结果到视频文件的线程
    std::string outputFilePath = "output.avi";
    ResultSaver resultSaver(resultQueue, outputFilePath, fps, frameSize);
    std::thread saveThread(resultSaver);

    // 等待所有线程完成
    videoThread.join();
    processThread.join();
    saveThread.join();

    return 0;
}
