#include <opencv2/opencv.hpp>
#include "PassportScanner.h"

int main() {
    const cv::Mat image = cv::imread("../images/z1.png");
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    cv::Mat processed_image = PassportScanner::preprocess(image);

    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Processed Image", cv::WINDOW_NORMAL);

    cv::imshow("Processed Image", processed_image);
    cv::imshow("Original Image", image);


    while (cv::waitKey(30) != 27);

    return 0;
}