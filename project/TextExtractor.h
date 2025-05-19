#pragma once

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <string>
#include <map>

class TextExtractor {
public:
    TextExtractor();
    ~TextExtractor();

    bool initialize(const std::string& language = "eng");

    [[nodiscard]] std::string extractText(const cv::Mat& image) const;

    [[nodiscard]] std::map<std::string, std::string> extractPassportInfo(const cv::Mat& image) const;

private:
    tesseract::TessBaseAPI* tessApi;
    bool initialized;

    static std::string parseMRZ(const std::string& text);
    static std::string parseField(const std::string& text, const std::string& fieldName);
};