#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

constexpr float MEAN[3][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1}
};

constexpr float GAUSSIAN[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

constexpr float LAPLACIAN[3][3] = {
    {0, -1, 0},
    {-1, 4, -1},
    {0, -1, 0}
};

constexpr float HIGHPASS[3][3] = {
    {-1, -1, -1},
    {-1, 8, -1},
    {-1, -1, -1}
};

enum FilterType {
    IDEAL_LOWPASS,
    IDEAL_HIGHPASS,
    GAUSSIAN_LOWPASS,
    GAUSSIAN_HIGHPASS
};

Mat applyFilter(const Mat& image, const float kernel[3][3], int kernelSize);
void centering_transform(Mat img);
Mat frequency_domain_filter(const Mat& src, FilterType filterType, int radius = 20, double A = 10);

int main() {
    const Mat image = imread("../images/cameraman.bmp", IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image" << std::endl;
        return -1;
    }

    const Mat meanImage = applyFilter(image, MEAN, 3);
    const Mat gaussianImage = applyFilter(image, GAUSSIAN, 3);
    const Mat laplacianImage = applyFilter(image, LAPLACIAN, 3);
    const Mat highPassImage = applyFilter(image, HIGHPASS, 3);

    imshow("Original Image", image);
    imshow("Mean Filter", meanImage);
    imshow("Gaussian Filter", gaussianImage);
    imshow("Laplace Filter", laplacianImage);
    imshow("High-pass Filter", highPassImage);

    while (waitKey(0) != 27) {}

    constexpr int radius = 20;
    constexpr double A = 10;

    Mat idealLowpass = frequency_domain_filter(image, IDEAL_LOWPASS, radius);
    Mat idealHighpass = frequency_domain_filter(image, IDEAL_HIGHPASS, radius);
    Mat gaussianLowpass = frequency_domain_filter(image, GAUSSIAN_LOWPASS, radius, A);
    Mat gaussianHighpass = frequency_domain_filter(image, GAUSSIAN_HIGHPASS, radius, A);

    imshow("Original Image", image);
    imshow("Ideal Low-pass Filter", idealLowpass);
    imshow("Ideal High-pass Filter", idealHighpass);
    imshow("Gaussian Low-pass Filter", gaussianLowpass);
    imshow("Gaussian High-pass Filter", gaussianHighpass);

    while (waitKey(0) != 27) {}
    return 0;
}

void centering_transform(Mat img) {
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
        }
    }
}

Mat frequency_domain_filter(const Mat& src, FilterType filterType, int radius, double A ) {
    int height = src.rows;
    int width = src.cols;

    Mat srcf;
    src.convertTo(srcf, CV_32FC1);

    centering_transform(srcf);

    Mat fourier;
    dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

    Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
    split(fourier, channels);

    Mat mag, phi;
    magnitude(channels[0], channels[1], mag);
    phase(channels[0], channels[1], phi);

    Mat magForDisplay = mag + 1;
    log(magForDisplay, magForDisplay);
    normalize(magForDisplay, magForDisplay, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("Magnitude Spectrum", magForDisplay);

    Mat filter = Mat::zeros(src.size(), CV_32F);
    Point center(width / 2, height / 2);

    for (int v = 0; v < height; v++) {
        for (int u = 0; u < width; u++) {
            double distance = sqrt(pow(u - center.x, 2) + pow(v - center.y, 2));

            switch (filterType) {
                case IDEAL_LOWPASS:
                    filter.at<float>(v, u) = (distance <= radius) ? 1.0f : 0.0f;
                    break;

                case IDEAL_HIGHPASS:
                    filter.at<float>(v, u) = (distance > radius) ? 1.0f : 0.0f;
                    break;

                case GAUSSIAN_LOWPASS:
                    filter.at<float>(v, u) = exp(-(distance * distance) / (2 * A * A));
                    break;

                case GAUSSIAN_HIGHPASS:
                    filter.at<float>(v, u) = 1 - exp(-(distance * distance) / (2 * A * A));
                    break;
            }
        }
    }

    Mat filterDisplay;
    normalize(filter, filterDisplay, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("Filter", filterDisplay);

    channels[0] = channels[0].mul(filter);
    channels[1] = channels[1].mul(filter);

    Mat dst, dstf;
    merge(channels, 2, fourier);
    dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

    centering_transform(dstf);

    normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

    return dst;
}

Mat applyFilter(const Mat& image, const float kernel[3][3], const int kernelSize) {
    Mat output = image.clone();
    const int pad = kernelSize / 2;

    bool hasNegativeValues = false;
    float sumPos = 0.0f, sumNeg = 0.0f;
    float sum = 0.0f;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            if (kernel[i][j] < 0) {
                hasNegativeValues = true;
                sumNeg -= kernel[i][j];
            } else {
                sumPos += kernel[i][j];
            }
            sum += kernel[i][j];
        }
    }

    for (int y = pad; y < image.rows - pad; y++) {
        for (int x = pad; x < image.cols - pad; x++) {
            float value = 0.0f;

            for (int ky = -pad; ky <= pad; ky++) {
                for (int kx = -pad; kx <= pad; kx++) {
                    value += kernel[ky + pad][kx + pad] * image.at<uchar>(y + ky, x + kx);
                }
            }

            if (hasNegativeValues) {
                constexpr float L = 255.0f;
                const float scaleFactor = L / (2 * std::max(sumPos, sumNeg));
                value = value * scaleFactor + L/2;
            } else {
                if (sum != 0) {
                    value /= sum;
                }
            }

            output.at<uchar>(y, x) = static_cast<uchar>(std::min(std::max(value, 0.0f), 255.0f));
        }
    }

    return output;
}