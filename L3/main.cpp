#include <iostream>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

void showHistogram(const string& name, const vector<int>& hist, const int hist_cols, const int hist_height) {
    Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
    // constructs a white image
    // computes histogram maximum
    int max_hist = 0;
    for (int i = 0; i < hist_cols; i++)
        if (hist[i] > max_hist)
            max_hist = hist[i];
    double scale = 1.0;
    scale = (double)hist_height / max_hist;
    int baseline = hist_height - 1;
    for (int x = 0; x < hist_cols; x++) {
        Point p1 = Point(x, baseline);
        Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
        line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins
    }
    imshow(name, imgHist);
}

vector<int> create_histogram(Mat &img) {
    vector hist(256, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            hist[img.at<uchar>(i, j)]++;
        }
    }
    return hist;
}

vector<int> create_histogram_bins(Mat& img, int m) {
    vector hist(m, 0);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            hist[img.at<uchar>(i, j) / m]++;
        }
    }
    return hist;
}

vector<float> compute_pdf(Mat &img) {
    vector<float> pdf(256, 0);
    vector<int> hist = create_histogram(img);

    for (int i = 0; i < 256; i++) {
        pdf[i] = (float)hist[i] / (img.rows * img.cols);
    }

    return pdf;
}

void show_treshold_image(Mat &img, const int window) {
    const vector<float> pdf = compute_pdf(img);
    vector<int> maxima;
    const int WH = window;
    const int window_size = 2 * WH + 1;

    for (int k = WH; k <= 255 - WH; ++k) {
        constexpr float TH = 0.0003;
        float sum = 0.0;
        for (int i = k - WH; i <= k + WH; ++i) {
            sum += pdf[i];
        }
        float v = sum / window_size;

        bool is_maximum = pdf[k] > v + TH;
        for (int i = k - WH; i <= k + WH && is_maximum; ++i) {
            if (pdf[k] < pdf[i]) {
                is_maximum = false;
            }
        }

        if (is_maximum) {
            maxima.push_back(k);
            k += WH;
        }
    }

    maxima.insert(maxima.begin(), 0);
    maxima.push_back(255);

    Mat thresholded_img = img.clone();

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int pixel_value = img.at<uchar>(i, j);
            int closest_max = maxima[0];
            int min_diff = abs(pixel_value - maxima[0]);

            for (size_t k = 1; k < maxima.size(); ++k) {
                int diff = abs(pixel_value - maxima[k]);
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_max = maxima[k];
                }
            }

            thresholded_img.at<uchar>(i, j) = closest_max;
        }
    }

    imshow("Thresholded Image", thresholded_img);
    waitKey(0);
}

void floyd_steinberg_dithering(Mat& img, const int window) {
    const vector<float> pdf = compute_pdf(img);
    vector<int> maxima;
    const int WH = window;
    const int window_size = 2 * WH + 1;

    for (int k = WH; k <= 255 - WH; ++k) {
        constexpr float TH = 0.0003;
        float sum = 0.0;
        for (int i = k - WH; i <= k + WH; ++i) {
            sum += pdf[i];
        }
        float v = sum / window_size;

        bool is_maximum = pdf[k] > v + TH;
        for (int i = k - WH; i <= k + WH && is_maximum; ++i) {
            if (pdf[k] < pdf[i]) {
                is_maximum = false;
            }
        }

        if (is_maximum) {
            maxima.push_back(k);
            k += WH;
        }
    }

    maxima.insert(maxima.begin(), 0);
    maxima.push_back(255);

    Mat new_img = img.clone();

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int pixel_value = new_img.at<uchar>(i, j);
            int closest_max = maxima[0];
            int min_diff = abs(pixel_value - maxima[0]);

            for (size_t k = 1; k < maxima.size(); ++k) {
                int diff = abs(pixel_value - maxima[k]);
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_max = maxima[k];
                }
            }

            const int error = pixel_value - closest_max;
            new_img.at<uchar>(i, j) = closest_max;
            if (j + 1 < img.cols) {
                new_img.at<uchar>(i, j + 1) = saturate_cast<uchar>(new_img.at<uchar>(i, j + 1) + error * 7 / 16);
            }
            if (i + 1 < img.rows && j - 1 >= 0) {
                new_img.at<uchar>(i + 1, j - 1) = saturate_cast<uchar>(new_img.at<uchar>(i + 1, j - 1) + error * 3 / 16);
            }
            if (i + 1 < img.rows) {
                new_img.at<uchar>(i + 1, j) = saturate_cast<uchar>(new_img.at<uchar>(i + 1, j) + error * 5 / 16);
            }
            if (i + 1 < img.rows && j + 1 < img.cols) {
                new_img.at<uchar>(i + 1, j + 1) = saturate_cast<uchar>(new_img.at<uchar>(i + 1, j + 1) + error * 1 / 16);
            }
        }
    }

    imshow("Floyd Steinberg Image", new_img);
    waitKey(0);
}

int main(int argc, char* argv[]) {
    Mat img = imread("../images/saturn.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Image not found!" << endl;
        return -1;
    }
    show_treshold_image(img, 20);
    floyd_steinberg_dithering(img, 20);

    return 0;
}