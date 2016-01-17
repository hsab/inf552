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

AllMatchPositionGenerator::AllMatchPositionGenerator(Mat &input_m, Mat &output_m, Mat &mask_m, double kk) {
    k = kk;
    input = input_m;
    output = output_m;
    mask = mask_m;
	std::cout << "input cols, rows: " << input.cols << ", " << input.rows << std::endl;
	std::cout << "output cols, rows: " << output.cols << ", " << output.rows << std::endl;
	std::cout << "mask cols, rows: " << mask.cols << ", " << mask.rows << std::endl;
	costs = Mat(output.rows + input.rows - 1, output.cols + input.cols - 1, CV_32F);
	std::cout << "costs cols, rows: " << costs.cols << ", " << costs.rows << std::endl;
    //probas = Mat(output.rows + input.rows - 1, output.cols + input.cols - 1, CV_32FC1);
    Scalar mean, stddev;
    meanStdDev(input, mean, stddev);
    sigma = stddev[0];
	std::cout <<"sigma: "<< sigma << "\n";
    std::random_device rd;
    dre = std::default_random_engine(rd());
}

//Squared Euclidian norm for a vector of 3 unsigned chars
float norm2(Vec3b v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void AllMatchPositionGenerator::compute_cost(Point t) {
	//std::cout << t << std::endl;
	double c = 0.0;
    long At = 0;
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            if (0 <= j+t.y && j+t.y < output.rows && 0 <= i+t.x && i+t.x < output.cols) {
				//if (mask.at<uchar>(i + t.x, j + t.y) == 255) {
				/*
				if  (t.x == 228 && t.y == -31){
				std::cout<< "i:" << i << "j" << j<<"\n";
				std::cout << At << "," << c << "," << i + t.x << "," << j + t.y;
				}
				//*/	
                    At++;
					c += norm2(input.at<Vec3b>(j, i) - output.at<Vec3b>(j + t.y,i + t.x));
                //}
            }
        }
    }
    if (c == 0.0) {
		costs.at<float>(t.y + input.rows - 1,t.x + input.cols - 1) = numeric_limits<float>::has_infinity ? numeric_limits<float>::infinity() : numeric_limits<float>::max();
    } else {
		costs.at<float>(t.y + input.rows - 1,t.x + input.cols - 1) = c / At;
    }
}

void AllMatchPositionGenerator::compute_costs() {
    for (int i = -input.cols+1; i < output.cols; i++) {
        for (int j = -input.rows+1; j < output.rows; j++) {
			compute_cost(Point(i, j));
        }
    }
}

int AllMatchPositionGenerator::setup_generator() {
	//proba isn't used!! changed to cost
    vector<float> init_array;
    for (int i = 0; i < costs.cols; i++) {
        for (int j = 0; j < costs.rows; j++) {
            init_array.push_back(exp(-costs.at<float>(j, i) / (k * sigma * sigma)));
        }
    }
    //discrete_distribution<> dist(init_array.begin(), init_array.end());	//visual c++ 2013 doesn't have this constructor... LOL
	std::size_t ii(0);
//	discrete_distribution<> dist(init_array.size(), 0., 1., [&init_array, &ii](float){return init_array[ii++];} );	//work around
    distrib = dist;
    
    return init_array.size();
}

void AllMatchPositionGenerator::change_position(int &posX, int &posY) {
    compute_costs();
    setup_generator();
    int pos = distrib(dre);
    posX = pos / (costs.rows) - input.cols+1;
    posY = pos % (costs.rows) - input.rows+1;
	//std::cout << "change position done\n";
    //useless under release
	//assert(posX < output.cols);
    //assert(posY < output.rows);
}
