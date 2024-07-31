// Author: shaoshengsong
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "FrameQueue.h"
#include "VideoReader.h"
#include "FrameProcessor.h"
#include "ResultSaver.h"
#include "Inference.h"
#include "FrameResult.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    fs::path current_path = fs::current_path();
    std::string videoFilePath = (argc == 2) ? argv[1] : current_path.string()+"/1.mp4";
    std::cout << "videoFilePath " << videoFilePath << std::endl;
    int framesPerSecond = 1;

    ThreadSafeQueue<cv::Mat> frameQueue;
    ThreadSafeQueue<FrameResult> resultQueue;

    std::thread videoThread(VideoReader(videoFilePath, frameQueue, framesPerSecond));

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

    std::string projectBasePath =current_path.string() + "/ultralytics";
    bool runOnGPU = false;

    Inference inf(projectBasePath + "/yolov8s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    std::thread processThread(FrameProcessor(frameQueue, resultQueue, inf));

    std::string outputFilePath = "output.avi";
    std::thread saveThread(ResultSaver(resultQueue, outputFilePath, fps, frameSize));

    videoThread.join();
    processThread.join();
    saveThread.join();

    return 0;
}
