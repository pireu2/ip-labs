#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

using namespace cv;
using namespace std;

constexpr int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};

Point findStartingPoint(const Mat &img);
vector<int> computeDerivative(const vector<int> &chain);
void extractBorder();
void reconstructFromFile(const string &chainCodeFile, const string &backgroundFile);
void reconstructBorder(Mat &img, const Point &startPoint, const vector<int> &chainCode);

int main() {
    // extractBorder();
    reconstructFromFile("../images/reconstruct.txt", "../images/gray_background.bmp");
    return 0;
}

void reconstructBorder(Mat &img, const Point &startPoint, const vector<int> &chainCode) {
    Point currentPoint = startPoint;

    if (currentPoint.x >= 0 && currentPoint.x < img.cols &&
        currentPoint.y >= 0 && currentPoint.y < img.rows) {
        img.at<uchar>(currentPoint.y, currentPoint.x) = 0;
        }

    for (const int dir : chainCode) {
        currentPoint.x += dx[dir];
        currentPoint.y += dy[dir];

        if (currentPoint.x >= 0 && currentPoint.x < img.cols &&
            currentPoint.y >= 0 && currentPoint.y < img.rows) {
            img.at<uchar>(currentPoint.y, currentPoint.x) = 0;
            }
    }
}

void reconstructFromFile(const string &chainCodeFile, const string &backgroundFile) {
    Mat img = imread(backgroundFile, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Could not read the background image: " << backgroundFile << endl;
        return;
    }

    ifstream file(chainCodeFile);
    if (!file.is_open()) {
        cerr << "Could not open chain code file: " << chainCodeFile << endl;
        return;
    }

    int startX, startY;
    file >> startX >> startY;
    Point startPoint(startX, startY);

    string line;
    getline(file, line);

    int chainCodeCount;
    file >> chainCodeCount;

    getline(file, line);

    vector<int> chainCode;
    int code;
    while (file >> code) {
        chainCode.push_back(code);
    }
    file.close();

    cout << "Starting point: (" << startPoint.x << ", " << startPoint.y << ")" << endl;
    cout << "Chain code count from file: " << chainCodeCount << endl;
    cout << "Chain code length: " << chainCode.size() << endl;

    reconstructBorder(img, startPoint, chainCode);

    imshow("Reconstructed Border", img);
    waitKey(0);
}

void extractBorder() {
    Mat img = imread("../images/triangle_up.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Could not read the image." << endl;
        return;
    }

    Point const p0 = findStartingPoint(img);
    if (p0.x == -1) {
        cerr << "No object found." << endl;
        return;
    }

    cout << "Starting Point: " << p0 << endl;

    vector border = {p0};
    vector<int> chainCode;
    int currentDir = 5;
    Point pk = p0;

    do {
        int sDir;
        if (currentDir % 2 == 0) {
            sDir = (currentDir + 7) % 8;
        } else {
            sDir = (currentDir + 6) % 8;
        }

        int foundDir = -1;
        for (int i = 0; i < 8; ++i) {
            const int dir = (sDir + i) % 8;
            const int x = pk.x + dx[dir];
            const int y = pk.y + dy[dir];
            if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) {
                continue;
            }
            if (img.at<uchar>(y, x) != 255) {
                foundDir = dir;
                break;
            }
        }

        if (foundDir == -1) {
            cerr << "Error: Contour tracing failed." << endl;
            break;
        }

        chainCode.push_back(foundDir);
        pk.x += dx[foundDir];
        pk.y += dy[foundDir];
        border.push_back(pk);
        currentDir = foundDir;
    } while (!(border.size() > 1 && pk == p0));

    if (border.size() > 1 && pk == p0) {
        border.pop_back();
    }

    if (!chainCode.empty() && pk == p0) {
        chainCode.pop_back();
    }

    vector<int> derivative = computeDerivative(chainCode);

    Mat borderImg = Mat::zeros(img.size(), CV_8UC1);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            borderImg.at<uchar>(i, j) = 255;
        }
    }
    for (const Point &p: border) {
        if (p.x >= 0 && p.x < img.cols && p.y >= 0 && p.y < img.rows) {
            borderImg.at<uchar>(p.y, p.x) = 0;
        }
    }

    imshow("Border Image", borderImg);
    waitKey(0);

    cout << "Chain codes: ";
    for (int code: chainCode) {
        cout << code << " ";
    }
    cout << endl;

    cout << "Derivatives: ";
    for (const int d: derivative) {
        cout << d << " ";
    }
    cout << endl;
}


Point findStartingPoint(const Mat &img) {
    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            if (img.at<uchar>(y, x) != 255) {
                return {x, y};
            }
        }
    }
    return {-1, -1};
}

vector<int> computeDerivative(const vector<int> &chain) {
    vector<int> deriv;
    const int n = static_cast<int>(chain.size());
    if (n == 0) return deriv;

    for (int i = 1; i < n; ++i) {
        const int prev = (i - 1 + n) % n;
        int d = (chain[i] - chain[prev] + 8) % 8;
        deriv.push_back(d);
    }
    deriv.push_back((chain[0] - chain[n - 1] + 8) % 8);
    return deriv;
}

