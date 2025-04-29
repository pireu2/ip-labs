#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

constexpr int dx[4] = {-1, 0, 1, 0};
constexpr int dy[4] = {0, -1, 0, 1};

Mat dilate(const Mat& src);
void runDilate();
Mat erosion(const Mat& src);
void runErosion();
Mat opening(const Mat& src);
void runOpening();
Mat closing(const Mat& src);
void runClosing();
Mat boundaryExtraction(const Mat& src);
void runBoundaryExtraction();
Mat regionFilling(const Mat& src);
void runRegionFilling();

Mat runNTimes(const Mat& src, unsigned int n, Mat (*func)(const Mat&));


int main() {
    runDilate();
    runErosion();
    runOpening();
    runClosing();
    runBoundaryExtraction();
    runRegionFilling();
    return 0;
}

void runRegionFilling() {
    const Mat src = imread("../images/6_RegionFilling/reg1neg1_bw.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }

    const Mat filled = regionFilling(src);
    imshow("Original", src);
    imshow("Region Filled", filled);
    while (waitKey(0) & 0xFF != 27) {}
}

Mat regionFilling(const Mat& src) {
    auto Xk = Mat(src.size(), src.type(), Scalar(255));
    Mat Xk_prev;

    auto src_complement = Mat(src.size(), src.type());
    bitwise_not(src, src_complement);

    const int centerX = src.cols / 2;
    const int centerY = src.rows / 2;
    Xk.at<uchar>(centerY, centerX) = 0;

    do {
        Xk_prev = Xk.clone();

        auto dilated = Mat(src.size(), src.type(), Scalar(255));
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                if (Xk_prev.at<uchar>(y, x) == 0) {
                    dilated.at<uchar>(y, x) = 0;

                    for (int i = 0; i < 4; i++) {
                        const int newX = x + dx[i];
                        const int newY = y + dy[i];

                        if (newX >= 0 && newX < src.cols && newY >= 0 && newY < src.rows) {
                            dilated.at<uchar>(newY, newX) = 0;
                        }
                    }
                }
            }
        }

        Xk = Mat(src.size(), src.type(), Scalar(255));
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                if (dilated.at<uchar>(y, x) == 0 && src_complement.at<uchar>(y, x) == 0) {
                    Xk.at<uchar>(y, x) = 0;
                }
            }
        }

    } while (countNonZero(Xk != Xk_prev) > 0);


    return Xk;
}


void runBoundaryExtraction() {
    const Mat src = imread("../images/5_BoundaryExtraction/reg1neg1_bw.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }

    const Mat extracted = boundaryExtraction(src);
    imshow("Original", src);
    imshow("Boundary Extracted", extracted);
    while (waitKey(0) & 0xFF != 27) {}
}


Mat boundaryExtraction(const Mat& src) {
    Mat eroded = erosion(src);

    auto result = Mat(src.size(), src.type(), Scalar(255));

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<uchar>(y, x) == 0 && eroded.at<uchar>(y, x) != 0) {
                result.at<uchar>(y, x) = 0;
            }
        }
    }

    return result;
}


void runOpening() {
    const Mat src = imread("../images/3_Open/cel4thr3_bw.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }
    const Mat opened = opening(src);
    imshow("Original", src);
    imshow("Opened", opened);
    while (waitKey(0) & 0xFF != 27) {}
}

void runClosing() {
    const Mat src = imread("../images/4_Close/phn1thr1_bw.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }
    const Mat closed = runNTimes(src, 5, closing);
    imshow("Original", src);
    imshow("Closed", closed);
    while (waitKey(0) & 0xFF != 27) {}
}

Mat opening(const Mat& src) {
    return dilate(erosion(src));
}

Mat closing(const Mat &src) {
    return erosion(dilate(src));
}


Mat runNTimes(const Mat& src, unsigned int n, Mat (*func)(const Mat&)) {
    Mat result = src.clone();
    for (unsigned int i = 0; i < n; ++i) {
        result = func(result);
    }
    return result;
}

Mat erosion(const Mat& src) {
    auto result = Mat(src.size(), src.type(), Scalar(255));

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<uchar>(y, x) == 0) {
                bool allNeighborsBlack = true;

                for (int i = 0; i < 4; i++) {
                    const int newX = x + dx[i];
                    const int newY = y + dy[i];

                    if (newX < 0 || newX >= src.cols || newY < 0 || newY >= src.rows ||
                        src.at<uchar>(newY, newX) == 255) {
                        allNeighborsBlack = false;
                        break;
                        }
                }

                if (allNeighborsBlack) {
                    result.at<uchar>(y, x) = 0;
                }
            }
        }
    }

    return result;
}


void runErosion() {
    const Mat src = imread("../images/2_Erode/mon1thr1_bw.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }

    const Mat eroded = runNTimes(src, 10, erosion);
    imshow("Original", src);
    imshow("Eroded", eroded);
    while (waitKey(0) & 0xFF != 27) {}
}


Mat dilate(const Mat& src){
    auto result = Mat(src.size(), src.type(), Scalar(255));

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (src.at<uchar>(y, x) == 0) {
                for (int i = 0; i < 4; i++) {
                    const int newX = x + dx[i];
                    const int newY = y + dy[i];

                    if (newX >= 0 && newX < src.cols && newY >= 0 && newY < src.rows) {
                        result.at<uchar>(newY, newX) = 0;
                    }
                }
            }
        }
    }

    return result;
}

void runDilate() {
    const Mat src = imread("../images/1_Dilate/wdg2ded1_bw.bmp", IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return;
    }
    const Mat dilated = dilate(src);
    imshow("Original", src);
    imshow("Dilated", dilated);
    while (waitKey(0) & 0xFF != 27) {}
}
