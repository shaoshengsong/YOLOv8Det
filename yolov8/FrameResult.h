// Author: shaoshengsong
#ifndef FRAMERESULT_H
#define FRAMERESULT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "inference.h"

struct FrameResult {
    cv::Mat frame;
    std::vector<Detection> detections;
};

#endif // FRAMERESULT_H
