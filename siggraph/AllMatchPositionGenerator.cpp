//
//  AllMatchPositionGenerator.cpp
//  siggraph
//
//  Created by Camille MASSET on 08/01/2016.
//
//

#include <random>
#include <vector>
#include <iostream>
#include "AllMatchPositionGenerator.hpp"

using namespace std;
using namespace cv;

AllMatchPositionGenerator::AllMatchPositionGenerator(Mat &input, Mat &output, Mat &mask, double k) {
    k = k;
    input = input;
    output = output;
    mask = mask;
    costs = Mat(output.rows + input.rows - 1, output.cols + input.cols - 1, CV_32FC1);
    probas = Mat(output.rows + input.rows - 1, output.cols + input.cols - 1, CV_32FC1);
    Scalar mean, stddev;
    meanStdDev(input, mean, stddev);
    sigma = stddev[0];
    std::random_device rd;
    dre = std::default_random_engine(rd());
}

//Squared Euclidian norm for a vector of 3 unsigned chars
float norm2(Vec3b v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void AllMatchPositionGenerator::compute_cost(Point t) {
    float c = 0.0;
    int At = 0;
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            if (0 <= j+t.y && j+t.y < output.rows && 0 <= i+t.x && i+t.x < output.cols) {
                if (mask.at<uchar>(j+t.y, i+t.x) == 255) {
                    At++;
                    c += norm2(input.at<Vec3b>(j, i) - output.at<Vec3b>(j+t.y, i+t.x));
                }
            }
        }
    }
    
    costs.at<float>(t) = c / At;
}

void AllMatchPositionGenerator::compute_costs() {
    for (int i = -input.cols+1; i < output.cols+input.cols-1; i++) {
        for (int j = -input.rows+1; j < output.rows+input.rows-1; j++) {
            compute_cost(Point(i, j));
        }
    }
}

void AllMatchPositionGenerator::setup_generator() {
    vector<float> init_array;
    for (int i = 0; i < probas.cols; i++) {
        for (int j = 0; j < probas.rows; j++) {
            init_array.push_back(exp(-probas.at<float>(j, i) / (k * sigma * sigma)));
        }
    }
    discrete_distribution<> dist(init_array.begin(), init_array.end());
    distrib = dist;
}

Point AllMatchPositionGenerator::get_next_translation() {
    compute_costs();
    setup_generator();
    int pos = distrib(dre);
    int j = pos % probas.rows;
    int i = pos / probas.rows;
    cout << "(" << i << ", " << j << ")" << endl;
    
    return Point(i, j);
}
