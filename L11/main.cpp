#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

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

cv::Mat apply2DGaussianFilter(const cv::Mat& src) {
    const cv::Mat kernel = createGaussianKernel2D(3, 0.5);
    cv::Mat dst;

    cv::filter2D(src, dst, -1, kernel);
    return dst;
}


cv::Mat cannyEdgeDetection(const cv::Mat& inputImage) {
    cv::Mat blurred = apply2DGaussianFilter(inputImage);

    cv::Mat displayBlurred = blurred.clone();

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


    cv::Mat magnitude = cv::Mat::zeros(blurred.size(), CV_8UC1);
    cv::Mat direction = cv::Mat::zeros(blurred.size(), CV_32F);

    for (int y = 1; y < blurred.rows - 1; y++) {
        for (int x = 1; x < blurred.cols - 1; x++) {
            int gx = 0;
            int gy = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    const int kernel_idx = (ky + 1) * 3 + (kx + 1);
                    const uchar pixel_value = blurred.at<uchar>(y + ky, x + kx);

                    gx += Gx[kernel_idx] * pixel_value;
                    gy += Gy[kernel_idx] * pixel_value;
                }
            }

            const int mag = static_cast<int>(std::sqrt(gx * gx + gy * gy));
            magnitude.at<uchar>(y, x) = static_cast<uchar>(mag);
            auto theta = static_cast<float>(std::atan2(gy, gx));
            direction.at<float>(y, x) = theta + static_cast<float>(CV_PI);
        }
    }

    magnitude.convertTo(magnitude, CV_8UC1, 255.0 / (255 * 4 * sqrt(4)));

    cv::Mat displayGradMagnitude = magnitude.clone();

    cv::Mat suppressed = cv::Mat::zeros(blurred.size(), CV_8UC1);

    for (int y = 1; y < blurred.rows - 1; y++) {
        for (int x = 1; x < blurred.cols - 1; x++) {
            float theta = direction.at<float>(y, x);

            int dir = 0;
            if ((theta > CV_PI/8 && theta <= 3*CV_PI/8) ||
                (theta > 9*CV_PI/8 && theta <= 11*CV_PI/8)) {
                dir = 1; // 45 degrees
            } else if ((theta > 3*CV_PI/8 && theta <= 5*CV_PI/8) ||
                      (theta > 11*CV_PI/8 && theta <= 13*CV_PI/8)) {
                dir = 0; // vertical
            } else if ((theta > 5*CV_PI/8 && theta <= 7*CV_PI/8) ||
                      (theta > 13*CV_PI/8 && theta <= 15*CV_PI/8)) {
                dir = 3; // 135 degrees
            } else {
                dir = 2; // horizontal
            }

            uchar val = magnitude.at<uchar>(y, x);
            bool isMax = false;

            switch (dir) {
                case 0: // vertical
                    isMax = (val >= magnitude.at<uchar>(y-1, x) &&
                            val >= magnitude.at<uchar>(y+1, x));
                    break;
                case 1: // 45 degrees
                    isMax = (val >= magnitude.at<uchar>(y-1, x+1) &&
                            val >= magnitude.at<uchar>(y+1, x-1));
                    break;
                case 2: // horizontal
                    isMax = (val >= magnitude.at<uchar>(y, x-1) &&
                            val >= magnitude.at<uchar>(y, x+1));
                    break;
                case 3: // 135 degrees
                    isMax = (val >= magnitude.at<uchar>(y-1, x-1) &&
                            val >= magnitude.at<uchar>(y+1, x+1));
                    break;
            }

            suppressed.at<uchar>(y, x) = isMax ? val : 0;
        }
    }


    imshow("Blurred", displayBlurred);
    imshow("Gradient Magnitude", displayGradMagnitude);
    imshow("Non-Maximum Suppression", suppressed);

    while (cv::waitKey(0) != 27) {
        // Wait for the user to press 'ESC' to exit
    }

    return suppressed;
}

int main() {
    const cv::Mat image = cv::imread("../saturn.bmp", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return -1;
    }

    cannyEdgeDetection(image);




    return 0;
}