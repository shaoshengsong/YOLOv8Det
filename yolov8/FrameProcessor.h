// Author: shaoshengsong

#ifndef FRAMEPROCESSOR_H
#define FRAMEPROCESSOR_H

#include "FrameQueue.h"
#include "Inference.h"
#include "FrameResult.h"

class FrameProcessor {
public:
    FrameProcessor(ThreadSafeQueue<cv::Mat>& frameQueue, ThreadSafeQueue<FrameResult>& resultQueue, Inference& inf)
        : frameQueue(frameQueue), resultQueue(resultQueue), inf(inf) {}

    void operator()() {
        while (true) {
            cv::Mat frame;
            if (frameQueue.waitAndPop(frame)) {
                std::vector<Detection> output = inf.runInference(frame);
                FrameResult frameResult = { frame, output };
                resultQueue.push(frameResult);
            } else {
                break;
            }
        }
    }

private:
    ThreadSafeQueue<cv::Mat>& frameQueue;
    ThreadSafeQueue<FrameResult>& resultQueue;
    Inference& inf;
};

#endif // FRAMEPROCESSOR_H
