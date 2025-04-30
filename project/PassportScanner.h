#pragma once

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>


class PassportScanner {
public:
    static cv::Mat preprocess(const cv::Mat& image);
private:
    static cv::Mat convertToGrayscale(const cv::Mat& image);
    static cv::Mat gaussianBlur(const cv::Mat& image, int kernel_size = 5);
    static cv::Mat edgeDetection(const cv::Mat& image);
    static cv::Mat threshold(const cv::Mat& image);
    static cv::Mat dilate(const cv::Mat& image, int kernel_size = 3);
    static cv::Mat erode(const cv::Mat& image, int kernel_size = 2);
    static cv::Mat opening(const cv::Mat& image);
    static cv::Mat findDocumentContours(const cv::Mat& image, const cv::Mat& originalImage,
                                        std::vector<cv::Point>& documentContour);

    static std::vector<cv::Point2f> orderPoints(const std::vector<cv::Point>& pts);
    static cv::Mat fourPointTransform(const cv::Mat& image, const std::vector<cv::Point>& pts);
};


