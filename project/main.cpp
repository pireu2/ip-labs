#include <opencv2/opencv.hpp>
#include "PassportScanner.h"
#include "TextExtractor.h"
#include <iostream>

int main() {
    const cv::Mat image = cv::imread("../images/2.png");
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        return -1;
    }

    const cv::Mat processed_image = PassportScanner::preprocess(image);

    TextExtractor extractor;
    if (!extractor.initialize()) {
        std::cerr << "Failed to initialize Tesseract OCR" << std::endl;
        return -1;
    }

    std::string text = extractor.extractText(processed_image);
    std::cout << "Extracted Text: " << text << std::endl;

    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::namedWindow("Processed Image", cv::WINDOW_NORMAL);

    cv::imshow("Processed Image", processed_image);
    cv::imshow("Original Image", image);


    while (cv::waitKey(30) != 27);

    return 0;
}