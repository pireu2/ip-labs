#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

cv::Mat applyMedianFilter(const cv::Mat& src, const int filterSize) {
    cv::Mat dst = src.clone();
    const int halfSize = filterSize / 2;

    for (int i = halfSize; i < src.rows - halfSize; i++) {
        for (int j = halfSize; j < src.cols - halfSize; j++) {
            std::vector<uchar> neighborhood;

            for (int k = -halfSize; k <= halfSize; k++) {
                for (int l = -halfSize; l <= halfSize; l++) {
                    neighborhood.push_back(src.at<uchar>(i + k, j + l));
                }
            }

            std::ranges::sort(neighborhood);
            dst.at<uchar>(i, j) = neighborhood[neighborhood.size() / 2];
        }
    }

    return dst;
}

cv::Mat createGaussianKernel2D(const int size, const double sigma) {
    cv::Mat kernel(size, size, CV_32F);
    double sum = 0.0;
    const int halfSize = size / 2;

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            const double x = i * i;
            const double y = j * j;
            const auto value = static_cast<float>(exp(-(x + y) / (2 * sigma * sigma)));
            kernel.at<float>(i + halfSize, j + halfSize) = value;
            sum += value;
        }
    }

    kernel /= sum;
    return kernel;
}

cv::Mat apply2DGaussianFilter(const cv::Mat& src,const int filterSize) {
    const double sigma = filterSize / 6.0;
    const cv::Mat kernel = createGaussianKernel2D(filterSize, sigma);
    cv::Mat dst;

    cv::filter2D(src, dst, -1, kernel);
    return dst;
}

std::pair<cv::Mat, cv::Mat> createSeparableGaussianKernels(const int size,const  double sigma) {
    cv::Mat kernelX = cv::Mat::zeros(1, size, CV_32F);
    cv::Mat kernelY = cv::Mat::zeros(size, 1, CV_32F);
    double sumX = 0.0, sumY = 0.0;
    const int halfSize = size / 2;

    for (int i = -halfSize; i <= halfSize; i++) {
        const auto value = static_cast<float>(exp(-(i * i) / (2 * sigma * sigma)));
        kernelX.at<float>(0, i + halfSize) = value;
        kernelY.at<float>(i + halfSize, 0) = value;
        sumX += value;
        sumY += value;
    }

    kernelX /= sumX;
    kernelY /= sumY;

    return {kernelX, kernelY};
}

cv::Mat applySeparableGaussianFilter(const cv::Mat& src,const int filterSize) {
    const double sigma = filterSize / 6.0;
    auto [kernelX, kernelY] = createSeparableGaussianKernels(filterSize, sigma);

    cv::Mat temp, dst;
    cv::filter2D(src, temp, -1, kernelX);
    cv::filter2D(temp, dst, -1, kernelY);

    return dst;
}

int main() {
    const cv::Mat saltPepperImage = cv::imread("../images/balloons_Salt&Pepper.bmp", cv::IMREAD_GRAYSCALE);
    const cv::Mat gaussianNoiseImage = cv::imread("../images/portrait_Gauss2.bmp", cv::IMREAD_GRAYSCALE);

    std::vector<int> filterSizes = {3, 5, 7};

    std::cout << "Exercise 1: Median Filter\n";
    for (const int size : filterSizes) {
        auto t = static_cast<double>(cv::getTickCount());
        cv::Mat medianResult = applyMedianFilter(saltPepperImage, size);
        t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
        std::cout << "Median filter (size " << size << "x" << size << "): Time = " << t * 1000 << " [ms]\n";

        cv::imshow("Salt & Pepper Noise", saltPepperImage);
        cv::imshow("Median Filter " + std::to_string(size), medianResult);
        while (cv::waitKey(0) != 27);
    }

    std::cout << "\nExercise 2: 2D Gaussian Filter\n";
    for (const int size : filterSizes) {
        auto t = static_cast<double>(cv::getTickCount());
        cv::Mat gaussianResult = apply2DGaussianFilter(gaussianNoiseImage, size);
        t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
        std::cout << "2D Gaussian filter (size " << size << "x" << size << "): Time = " << t * 1000 << " [ms]\n";

        cv::imshow("Gaussian Noise", gaussianNoiseImage);
        cv::imshow("2D Gaussian Filter " + std::to_string(size), gaussianResult);
        while (cv::waitKey(0) != 27);
    }

    std::cout << "\nExercise 3: Separable Gaussian Filter\n";
    for (const int size : filterSizes) {
        auto t = static_cast<double>(cv::getTickCount());
        cv::Mat sepGaussianResult = applySeparableGaussianFilter(gaussianNoiseImage, size);
        t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
        std::cout << "Separable Gaussian filter (size " << size << "x" << size << "): Time = " << t * 1000 << " [ms]\n";

        cv::imshow("Gaussian Noise", gaussianNoiseImage);
        cv::imshow("Separable Gaussian Filter " + std::to_string(size), sepGaussianResult);
        while (cv::waitKey(0) != 27);
    }

    cv::destroyAllWindows();
    return 0;
}