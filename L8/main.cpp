#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void showHistogram(const string& name, const vector<int>& hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
    int max_hist = 0;
    for (int i = 0; i < hist_cols; i++)
        if (hist[i] > max_hist)
            max_hist = hist[i];
    const double scale = static_cast<double>(hist_height) / max_hist;
    const int baseline = hist_height - 1;
    for (int x = 0; x < hist_cols; x++) {
        const auto p1 = Point(x, baseline);
        const auto p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(255, 0, 255));
    }
    imshow(name, imgHist);
    waitKey(0);
}

void computeImageStats(const Mat& img) {
    vector histogram(256,0);
    vector cumHistogram(256,0);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            histogram[img.at<uchar>(i, j)]++;
        }
    }

    cumHistogram[0] = histogram[0];
    for (int i = 1; i < histogram.size(); i++) {
        cumHistogram[i] = cumHistogram[i - 1] + histogram[i];
    }

    double mean = 0.0;
    const int M = img.rows * img.cols;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            mean += img.at<uchar>(i, j);
        }
    }
    mean /= M;

    double stdDev = 0.0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            stdDev += pow(img.at<uchar>(i, j) - mean, 2);
        }
    }
    stdDev /= M;
    stdDev = sqrt(stdDev);

    cout << "Image Statistics:\n";
    cout << "Mean: " << mean << "\n";
    cout << "Standard Deviation: " << stdDev << "\n";

    showHistogram("Histogram", histogram, 256, 200);
    showHistogram("Cumulative Histogram", cumHistogram, 256, 200);
}


Mat computeThreshold(const Mat& img) {
    vector histogram(256, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            histogram[img.at<uchar>(i, j)]++;
        }
    }

    int min_val = 255, max_val = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            const int val = img.at<uchar>(i, j);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    double T = (min_val + max_val) / 2.0;
    double T_prev;
    double error = 0.1;

    do {
        T_prev = T;

        int N1 = 0, N2 = 0;
        double sum1 = 0.0, sum2 = 0.0;

        for (int g = min_val; g <= T; g++) {
            sum1 += g * histogram[g];
            N1 += histogram[g];
        }

        for (int g = static_cast<int>(T) + 1; g <= max_val; g++) {
            sum2 += g * histogram[g];
            N2 += histogram[g];
        }

        const double mean_G1 = N1 > 0 ? sum1 / N1 : 0;
        const double mean_G2 = N2 > 0 ? sum2 / N2 : 0;

        T = (mean_G1 + mean_G2) / 2.0;

    } while (abs(T - T_prev) >= error);

    Mat binary = img.clone();
    threshold(img, binary, T, 255, THRESH_BINARY);

    cout << "Computed threshold: " << T << endl;

    return binary;
}

Mat histogramStretchShrink(const Mat& img, const int out_min, const int out_max) {
    Mat result = img.clone();
    int in_min = 255, in_max = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            const int val = img.at<uchar>(i, j);
            if (val < in_min) in_min = val;
            if (val > in_max) in_max = val;
        }
    }

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            double const val = img.at<uchar>(i, j);
            int new_val = out_min + (val - in_min) * (out_max - out_min) / (in_max - in_min);
            new_val = max(0, min(255, new_val));
            result.at<uchar>(i, j) = new_val;
        }
    }

    return result;
}

Mat gammaCorrection(const Mat& img, const double gamma) {
    Mat result = img.clone();

    unsigned char lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = saturate_cast<uchar>(pow((i / 255.0), gamma) * 255.0);
    }

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            result.at<uchar>(i, j) = lut[img.at<uchar>(i, j)];
        }
    }

    return result;
}

Mat histogramEqualization(const Mat& img) {
    Mat result = img.clone();
    const int M = img.rows * img.cols;

    vector histogram(256, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            histogram[img.at<uchar>(i, j)]++;
        }
    }

    vector pdf(256, 0.0);
    for (int i = 0; i < 256; i++) {
        pdf[i] = static_cast<double>(histogram[i]) / M;
    }

    vector cpdf(256, 0.0);
    cpdf[0] = pdf[0];
    for (int i = 1; i < 256; i++) {
        cpdf[i] = cpdf[i-1] + pdf[i];
    }

    vector<uchar> lut(256, 0);
    for (int i = 0; i < 256; i++) {
        lut[i] = saturate_cast<uchar>(255.0 * cpdf[i]);
    }

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            result.at<uchar>(i, j) = lut[img.at<uchar>(i, j)];
        }
    }

    return result;
}


int main() {
    const Mat img = imread("../images/Hawkes_Bay_NZ.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Could not open or find the image.\n";
        return -1;
    }

    imshow("Original Image", img);

    int outMin, outMax;
    double gamma;

    cout << "Enter min output value for stretching/shrinking (0-255): ";
    cin >> outMin;
    cout << "Enter max output value for stretching/shrinking (0-255): ";
    cin >> outMax;

    const Mat stretched = histogramStretchShrink(img, outMin, outMax);
    imshow("Stretched/Shrunk Image", stretched);

    cout << "Enter gamma correction value (0.1-5.0): ";
    cin >> gamma;

    const Mat gammaCorrected = gammaCorrection(img, gamma);
    imshow("Gamma Corrected Image", gammaCorrected);

    const Mat equalized = histogramEqualization(img);
    imshow("Equalized Image", equalized);

    while (waitKey(0) & 0xFF != 27){}
    return 0;
}