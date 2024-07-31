// Author: shaoshengsong
#ifndef VIDEOREADER_H
#define VIDEOREADER_H

#include <opencv2/opencv.hpp>
#include "FrameQueue.h"

class VideoReader {
public:
    VideoReader(const std::string& videoFilePath, ThreadSafeQueue<cv::Mat>& frameQueue, int framesPerSecond)
        : videoFilePath(videoFilePath), frameQueue(frameQueue), framesPerSecond(framesPerSecond) {}

    void operator()() {
        cv::VideoCapture cap(videoFilePath);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open video file." << std::endl;
            return;
        }

        double fps = cap.get(cv::CAP_PROP_FPS);
        int frameInterval = static_cast<int>(fps / framesPerSecond);

        cv::Mat frame;
        int frameCount = 0;

        while (true) {
            cap >> frame;
            if (frame.empty()) {
                break;
            }

            if (frameCount % frameInterval == 0) {
                frameQueue.push(frame.clone());
                std::cout << "Frame " << frameCount << " added to queue." << std::endl;
            }

            frameCount++;
        }

        cap.release();
        std::cout << "Video reading completed. Total frames added to queue: " << frameCount / frameInterval << std::endl;
    }

private:
    std::string videoFilePath;
    ThreadSafeQueue<cv::Mat>& frameQueue;
    int framesPerSecond;
};

#endif // VIDEOREADER_H
