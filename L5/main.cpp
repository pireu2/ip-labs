#include <iostream>
#include <vector>
#include <queue>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


Mat bfs(const Mat &img);
std::pair<Mat,Mat> twoPass(const Mat &img);
Mat displayComponents(const Mat& labels, int numLabels);

int main() {
    const Mat img = imread("../images/letters.bmp", IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Image not found" << std::endl;
        return 1;
    }

    imshow("img", img);
    waitKey(0);

    Mat bfsLabels = bfs(img);
    auto [firstPass, secondPass] = twoPass(img);

    const int numLabelsBfs = *std::max_element(bfsLabels.begin<int>(), bfsLabels.end<int>());
    const int numLabelsFirstPass = *std::max_element(firstPass.begin<int>(), firstPass.end<int>());
    const int numLabelsSecondPass = *std::max_element(secondPass.begin<int>(), secondPass.end<int>());

    const Mat bfsColored = displayComponents(bfsLabels, numLabelsBfs);
    const Mat firstPassColored = displayComponents(firstPass, numLabelsFirstPass);
    const Mat secondPassColored = displayComponents(secondPass, numLabelsSecondPass);

    imshow("bfs", bfsColored);
    waitKey(0);

    imshow("firstPass", firstPassColored);
    waitKey(0);

    imshow("secondPass", secondPassColored);
    waitKey(0);

    return 0;
}

std::pair<Mat,Mat> twoPass(const Mat &img) {
    int label = 0;
    Mat labels = Mat::zeros(img.size(), CV_32S);
    auto edges = std::vector<std::vector<int>>(1000);

    constexpr int dx[] = {-1, -1, -1, 0};
    constexpr int dy[] = {-1, 0, 1, -1};

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i,j) == 0 && labels.at<int>(i,j) == 0) {
                auto L = std::vector<int>();

                for (int k = 0; k < 4; k++) {
                    const int x = i + dx[k];
                    const int y = j + dy[k];
                    if (x >= 0 && x < img.rows && y >= 0 && y < img.cols) {
                        if (labels.at<int>(x, y) > 0) {
                            L.push_back(labels.at<int>(x, y));
                        }
                    }
                }

                if (L.empty()) {
                    label++;
                    labels.at<int>(i, j) = label;
                }
                else {
                    const int min = *std::ranges::min_element(L);
                    labels.at<int>(i, j) = min;
                    for (const auto &l : L) {
                        if (l != min) {
                            edges[l].push_back(min);
                            edges[min].push_back(l);
                        }
                    }
                }
            }
        }
    }

    const Mat firstPass = labels.clone();

    int newLabel = 0;
    auto newLabels = std::vector(label + 1, 0);

    for (int i = 1; i < label; i++) {
        if (newLabels[i] == 0) {
            newLabel++;
            std::queue<int> q;
            newLabels[i] = newLabel;
            q.push(i);
            while (!q.empty()) {
                const auto x = q.front();
                q.pop();
                for (const auto &edge : edges[x]) {
                    if (newLabels[edge] == 0) {
                        newLabels[edge] = newLabel;
                        q.push(edge);
                    }
                }
            }
        }
    }

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            labels.at<int>(i, j) = newLabels[labels.at<int>(i, j)];
        }
    }

    return {firstPass, labels};
}

Mat bfs(const Mat &img) {
    int label = 0;
    Mat labels = Mat::zeros(img.size(), CV_32S);

    constexpr int dx[] = {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr int dy[] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
                label++;
                std::queue<std::pair<int, int>> q;
                labels.at<int>(i, j) = label;
                q.emplace(i, j);

                while (!q.empty()) {
                    auto [fst, snd] = q.front();
                    q.pop();

                    for (int k = 0; k < 8; k++) {
                        const int x = fst + dx[k];
                        const int y = snd + dy[k];
                        if (x >= 0 && x < img.rows && y >= 0 && y < img.cols) {
                            if (img.at<uchar>(x, y) == 0 && labels.at<int>(x, y) == 0) {
                                labels.at<int>(x, y) = label;
                                q.emplace(x, y);
                            }
                        }
                    }
                }
            }
        }
    }

    return labels;
}

Mat displayComponents(const Mat& labels, int numLabels) {
    std::vector<Vec3b> colors(numLabels + 1);
    colors[0] = Vec3b(255, 255, 255);

    for (int i = 1; i <= numLabels; i++) {
        colors[i] = Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }

    Mat colored = Mat::zeros(labels.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; i++) {
        for (int j = 0; j < labels.cols; j++) {
            colored.at<Vec3b>(i, j) = colors[labels.at<int>(i, j)];
        }
    }

    return colored;
}