#include <iostream>
#include <cmath>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void onMouse(int event, int x, int y, int flags, void *userdata);
int calcArea(Mat &img, const Vec3b &color);
Point2i calcCenterOfMass(Mat &img, int area, const Vec3b &color);
double calcAngleOfElongation(Mat &img, const Point2i &centerOfMass, const Vec3b &color);
std::pair<int, int> findMinMaxColumns(Mat &img, const Vec3b &color);
int calcPerimeter(Mat &img, const Vec3b &color);
float calcThinnessRatio(int perimeter, int area);
float calcAspectRatio(Mat &img, const Vec3b &color);
void showProjections(Mat &img, const Vec3b &color);

int main(int argc, char **argv) {
    Mat image = imread("../images/trasaturi_geom.bmp", IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return -1;
    }
    namedWindow("Image", WINDOW_AUTOSIZE);

    setMouseCallback("Image", onMouse, &image);
    imshow("Image", image);
    while ((waitKey(0) & 0xFF) != 27){}
    return 0;
}

void onMouse(int event, int x, int y, int flags, void *userdata) {
    const auto image = static_cast<Mat *>(userdata);
    if (event == EVENT_LBUTTONDOWN) {
        if (x >= 0 && y >= 0 && x < image->cols && y < image->rows) {
            const Vec3b &color = image->at<Vec3b>(y, x);
            const auto objArea = calcArea(*image, color);
            const auto centerOfMass = calcCenterOfMass(*image, objArea, color);
            const auto phi = calcAngleOfElongation(*image, centerOfMass, color);
            const auto perimeter = calcPerimeter(*image, color);
            const auto thinnessRatio = calcThinnessRatio(perimeter, objArea);
            const auto aspectRatio = calcAspectRatio(*image, color);

            std::cout << "Area: " << objArea << "\n"
                  << "Center of mass: (" << centerOfMass.x << ", " << centerOfMass.y << ")\n"
                  << "Angle of elongation: " << static_cast<int>((phi + M_PI) * 180 / M_PI) << "\n"
                  << "Perimeter: " << perimeter << "\n"
                  << "Thinness ratio: " << thinnessRatio << "\n"
                  << "Aspect ratio: " << aspectRatio << "\n";

            Mat displayImage = image->clone();
            circle(displayImage, Point(centerOfMass.y, centerOfMass.x), 5, Scalar(0, 0, 255), -1);
            const auto [minCol, maxCol] = findMinMaxColumns(*image, color);
            const int r1 = static_cast<int>(centerOfMass.x + tan(phi) * (maxCol - centerOfMass.y));
            const int r2 = static_cast<int>(centerOfMass.x - tan(phi) * (centerOfMass.y - minCol));
            line(displayImage, Point(minCol, r2), Point(maxCol, r1), Scalar(0, 255, 0), 2);
            imshow("Image", displayImage);

            showProjections(*image, color);
        }
    }
}

int calcArea(Mat &img, const Vec3b &color) {
    int area = 0;
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<Vec3b>(i, j) == color) {
                area++;
            }
        }
    }
    return area;
}

Point2i calcCenterOfMass(Mat &img, const int area, const Vec3b &color) {
    int xSum = 0;
    int ySum = 0;
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            if (img.at<Vec3b>(r, c) == color) {
                ySum += c;
                xSum += r;
            }
        }
    }
    return {xSum / area, ySum / area};
}

double calcAngleOfElongation(Mat &img, const Point2i &centerOfMass, const Vec3b &color) {
    int mrc = 0;
    int mcc = 0;
    int mrr = 0;

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            if (img.at<Vec3b>(r, c) == color) {
                mrr += (r - centerOfMass.x) * (r - centerOfMass.x);
                mcc += (c - centerOfMass.y) * (c - centerOfMass.y);
                mrc += (r - centerOfMass.x) * (c - centerOfMass.y);
            }
        }
    }

    const double phi = atan2(2 * mrc, mcc - mrr) / 2;
    return phi;
}

std::pair<int, int> findMinMaxColumns(Mat &img, const Vec3b &color) {
    int minCol = img.cols - 1;
    int maxCol = 0;
    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            if (img.at<Vec3b>(r, c) == color) {
                minCol = std::min(minCol, c);
                maxCol = std::max(maxCol, c);
            }
        }
    }
    return {minCol, maxCol};
}

int calcPerimeter(Mat &img, const Vec3b &color) {
    int perimeter = 0;
    constexpr int dr[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    constexpr int dc[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    bool isContour = false;

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            isContour = false;
            if (img.at<Vec3b>(r, c) == color) {
                for (int k = 0; k < 8; k++) {
                    const int newR = r + dr[k];
                    const int newC = c + dc[k];

                    if (newR >= 0 && newR < img.rows && newC >= 0 && newC < img.cols) {
                        if (img.at<Vec3b>(newR, newC) != color) {
                            isContour = true;
                            break;
                        }
                    } else {
                        isContour = true;
                        break;
                    }
                }
            }
            if (isContour) {
                perimeter++;
            }
        }
    }

    return static_cast<int>(perimeter * M_PI / 4.0);
}

float calcThinnessRatio(const int perimeter, const int area) {
    return M_PI * 4 * static_cast<float>(area) / static_cast<float>(perimeter * perimeter);
}

float calcAspectRatio(Mat &img, const Vec3b &color) {
    int maxCol = 0, maxRow = 0;
    int minCol = img.cols - 1;
    int minRow = img.rows - 1;

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            if (img.at<Vec3b>(r, c) == color) {
                minRow = std::min(minRow, r);
                maxRow = std::max(maxRow, r);
                minCol = std::min(minCol, c);
                maxCol = std::max(maxCol, c);
            }
        }
    }

    const auto width = static_cast<float>(maxCol - minCol + 1);
    const auto height = static_cast<float>(maxRow - minRow + 1);

    return width / height;
}

void showProjections(Mat &img, const Vec3b &color) {
    std::vector horizontalProj(img.rows, 0);
    std::vector verticalProj(img.cols, 0);

    for (int r = 0; r < img.rows; r++) {
        for (int c = 0; c < img.cols; c++) {
            if (img.at<Vec3b>(r, c) == color) {
                horizontalProj[r]++;
                verticalProj[c]++;
            }
        }
    }

    const int maxHorz = *std::ranges::max_element(horizontalProj);
    const int maxVert = *std::ranges::max_element(verticalProj);

    constexpr int width = 500;
    constexpr int height = 300;

    Mat horzProjImg(height, width, CV_8UC3, Scalar(255, 255, 255));

    Mat vertProjImg(height, width, CV_8UC3, Scalar(255, 255, 255));

    for (int r = 0; r < img.rows; r++) {
        const int scaledRow = r * height / img.rows;
        const int projLength = horizontalProj[r] * width / maxHorz;

        line(horzProjImg,
             Point(0, scaledRow),
             Point(projLength, scaledRow),
             Scalar(0, 0, 255), 2);
    }

    for (int c = 0; c < img.cols; c++) {
        const int scaledCol = c * width / img.cols;
        const int projLength = verticalProj[c] * height / maxVert;

        line(vertProjImg,
             Point(scaledCol, height - 1),
             Point(scaledCol, height - 1 - projLength),
             Scalar(0, 0, 255), 2);
    }

    putText(horzProjImg, "Horizontal Projection", Point(10, 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1);

    putText(vertProjImg, "Vertical Projection", Point(10, 20),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 0), 1);

    imshow("Horizontal Projection", horzProjImg);
    imshow("Vertical Projection", vertProjImg);
}
