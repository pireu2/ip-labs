#include "PassportScanner.h"

cv::Mat PassportScanner::convertToGrayscale(const cv::Mat& image) {
    cv::Mat output(image.size(), CV_8UC1);
    for (int r = 0; r < image.rows; r++) {
        for (int c = 0; c < image.cols; c++) {
            const auto& pixel = image.at<cv::Vec3b>(r, c);
            output.at<uchar>(r, c) = 0.299*pixel[2] + 0.587*pixel[1] + 0.114*pixel[0];
        }
    }
    return output;
}

cv::Mat PassportScanner::gaussianBlur(const cv::Mat& image, const int kernel_size) {
    cv::Mat output = image.clone();
    constexpr float kernel[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}
    };
    const int pad = kernel_size / 2;

    for (int y = pad; y < image.rows - pad; y++) {
        for (int x = pad; x < image.cols - pad; x++) {
            constexpr float sum = 256.0f;
            float value = 0.0f;
            for (int ky = -pad; ky <= pad; ky++) {
                for (int kx = -pad; kx <= pad; kx++) {
                    value += kernel[ky + pad][kx + pad] * image.at<uchar>(y + ky, x + kx);
                }
            }
            output.at<uchar>(y, x) = static_cast<uchar>(value / sum);
        }
    }

    return output;
}

cv::Mat PassportScanner::edgeDetection(const cv::Mat& image) {
    cv::Mat output(image.size(), CV_8UC1);

    constexpr int8_t Gx[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    constexpr int8_t Gy[] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };

    for (int y = 1; y < image.rows - 1; y++) {
        for (int x = 1; x < image.cols - 1; x++) {
            int gx = 0;
            int gy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    const int kernel_idx = (ky + 1) * 3 + (kx + 1);
                    const uchar pixel_value = image.at<uchar>(y + ky, x + kx);

                    gx += Gx[kernel_idx] * pixel_value;
                    gy += Gy[kernel_idx] * pixel_value;
                }
            }

            const int magnitude = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            output.at<uchar>(y, x) = static_cast<uchar>(std::min(magnitude, 255));
        }
    }

    return output;
}


cv::Mat PassportScanner::threshold(const cv::Mat& image) {
    cv::Mat output(image.size(), CV_8UC1);

    cv::threshold(image, output, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    return output;
}

cv::Mat PassportScanner::dilate(const cv::Mat& image, const int kernel_size) {
    cv::Mat output(image.size(), CV_8UC1);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::dilate(image, output, kernel, cv::Point(-1, -1), 2);

    return output;
}

cv::Mat PassportScanner::erode(const cv::Mat& image, const int kernel_size) {
    cv::Mat output(image.size(), CV_8UC1);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::erode(image, output, kernel);

    return output;
}

cv::Mat PassportScanner::opening(const cv::Mat &image) {
    cv::Mat output(image.size(), CV_8UC1);
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(image, output, cv::MORPH_OPEN, kernel);
    return output;
}

cv::Mat PassportScanner::findDocumentContours(const cv::Mat& image, const cv::Mat& originalImage,
                                            std::vector<cv::Point>& documentContour) {
    cv::Mat output = originalImage.clone();


    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image.clone(), contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::ranges::sort(contours, [](const std::vector<cv::Point>& c1,
                                   const std::vector<cv::Point>& c2) {
        return cv::contourArea(c1) > cv::contourArea(c2);
    });

    cv::Mat allContours = originalImage.clone();
    cv::drawContours(allContours, contours, -1, cv::Scalar(0, 255, 0), 2);

    bool found = false;
    for (const auto& contour : contours) {
        const double peri = cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.05 * peri, true);

        if (approx.size() == 4) {
            documentContour = approx;
            found = true;
            break;
        }
    }

    if (found) {
        const std::vector<std::vector<cv::Point>> docContours = {documentContour};
        cv::drawContours(output, docContours, -1, cv::Scalar(0, 255, 0), 3);
    }

    return output;
}


std::vector<cv::Point2f> PassportScanner::orderPoints(const std::vector<cv::Point>& pts) {
    std::vector<cv::Point2f> rect(4);
    std::vector<cv::Point2f> points(4);

    for (int i = 0; i < 4; i++) {
        points[i] = cv::Point2f(static_cast<float>(pts[i].x), static_cast<float>(pts[i].y));
    }

    std::vector<float> sumValues(4);
    std::vector<float> diffValues(4);

    for (int i = 0; i < 4; i++) {
        sumValues[i] = points[i].x + points[i].y;
        diffValues[i] = points[i].y - points[i].x;
    }

    auto minSumIt = std::ranges::min_element(sumValues);
    auto maxSumIt = std::ranges::max_element(sumValues);

    rect[0] = points[std::distance(sumValues.begin(), minSumIt)];
    rect[2] = points[std::distance(sumValues.begin(), maxSumIt)];

    auto minDiffIt = std::ranges::min_element(diffValues);
    auto maxDiffIt = std::ranges::max_element(diffValues);

    rect[1] = points[std::distance(diffValues.begin(), minDiffIt)];
    rect[3] = points[std::distance(diffValues.begin(), maxDiffIt)];

    return rect;
}

cv::Mat PassportScanner::fourPointTransform(const cv::Mat& image, const std::vector<cv::Point>& pts) {
    const std::vector<cv::Point2f> rect = orderPoints(pts);

    const cv::Point2f tl = rect[0];
    const cv::Point2f tr = rect[1];
    const cv::Point2f br = rect[2];
    const cv::Point2f bl = rect[3];

    const float widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
    const float widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
    const int maxWidth = std::max(static_cast<int>(widthA), static_cast<int>(widthB));

    const float heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
    const float heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
    const int maxHeight = std::max(static_cast<int>(heightA), static_cast<int>(heightB));

    const std::vector<cv::Point2f> dst = {
        {0, 0},
        {static_cast<float>(maxWidth - 1), 0},
        {static_cast<float>(maxWidth - 1), static_cast<float>(maxHeight - 1)},
        {0, static_cast<float>(maxHeight - 1)}
    };

    const cv::Mat M = cv::getPerspectiveTransform(rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));

    return warped;
}

cv::Mat PassportScanner::preprocess(const cv::Mat &image) {
    cv::Mat gray_image = convertToGrayscale(image);
    cv::Mat blurred_image = gaussianBlur(gray_image);
    cv::Mat edge_image = edgeDetection(blurred_image);
    cv::Mat threshold_image = threshold(edge_image);
    cv::Mat dilated_image = dilate(threshold_image);
    cv::Mat eroded_image = erode(dilated_image);
    cv::Mat opening_image = opening(eroded_image);

    std::vector<cv::Point> document_contour;
    cv::Mat document_outline = findDocumentContours(opening_image, image, document_contour);
    if (document_contour.empty()) {
        return document_outline;
    }

    cv::Mat warped = fourPointTransform(image, document_contour);
    cv::Mat output = convertToGrayscale(warped);
    return output;
}





