// Author: shaoshengsong
#ifndef RESULTSAVER_H
#define RESULTSAVER_H

#include <opencv2/opencv.hpp>
#include "FrameQueue.h"
#include "FrameResult.h"

class ResultSaver {
public:
    ResultSaver(ThreadSafeQueue<FrameResult>& resultQueue, const std::string& outputFilePath, int fps, cv::Size frameSize)
        : resultQueue(resultQueue), outputFilePath(outputFilePath), fps(fps), frameSize(frameSize) {}

    void operator()() {
        cv::VideoWriter writer(outputFilePath, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not open the output video file for writing." << std::endl;
            return;
        }

        while (true) {
            FrameResult frameResult;
            if (resultQueue.waitAndPop(frameResult)) {
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
                break;
            }
        }

        writer.release();
        std::cout << "Video writing completed." << std::endl;
    }

private:
    ThreadSafeQueue<FrameResult>& resultQueue;
    std::string outputFilePath;
    int fps;
    cv::Size frameSize;
};

#endif // RESULTSAVER_H
