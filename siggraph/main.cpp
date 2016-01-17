//
//  main.cpp
//  siggraph
//
//  Created by Camille MASSET on 01/01/2016.
//  Copyright © 2016 Ecole polytechnique. All rights reserved.
// 

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <limits>
#include <cmath>
#include <algorithm>

#include "RandomPositionGenerator.h"
#include "AllMatchPositionGenerator.hpp"
#include "maxflow/graph.h"

using namespace std;
using namespace cv;

//enumeration for patch placement mathode selection
enum PatchPlacementMode {RANDOM, ALLMATCH, SUBPATCH};

//maximum possible value for graph cut
const double MAXVAL = numeric_limits<double>::has_infinity ? numeric_limits<double>::infinity() : numeric_limits<double>::max();

//Euclidian norm for a vector of 3 unsigned chars
double norm(Vec3b v) {
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

/*	cost function for 4 pixels
parameters:
Vec3b		vi1,vi2		pixels of the first image in position 1/2
Vec3b		vo1,vo2		pixels of the second image in position 1/2
*/
double cost(Vec3b vi1, Vec3b vi2, Vec3b vo1, Vec3b vo2) {
	return norm(vi1 - vo1) + norm(vi2 - vo2);
}

/*	cost function with position coordinates
parameters:
const Mat&		input,output	input image (pattern)/output image
int				iPosX,iPosY		position(upper-left corner) of the overlapping zone for input image
int				oPosX,oPosY		position(upper-left corner) of the overlapping zone for output image
int				x1,y1,x2,y2		coordinates of the first/second point in the overlapping zone
*/
double cost(const Mat& input, int iPosX, int iPosY, const Mat& output,int oPosX, int oPosY,
	int x1, int y1, int x2, int y2) {
	return cost(input.at<Vec3b>(y1 + iPosY, x1 + iPosX), input.at<Vec3b>(y2 + iPosY, x2 + iPosX),
		output.at<Vec3b>(y1 + oPosY, x1 + oPosX), output.at<Vec3b>(y2 + oPosY, x2 + oPosX));
}

/* Buggy for immense output size (output-patch > RAND_MAX), non-uniform distribution
//get patch and output size, generate random patch position (upper left corner pixel position)
void getRandomPosition(const int patchX, const int patchY, const int outputX, const int outputY, int& posX, int& posY){
	posX = rand() % (outputX - patchX + 1);
	posY = rand() % (outputY - patchY + 1);
}
*/

//structure registering cut information
struct Cut {
	//cost of cut
	double cost;
	//first(left/up) hidden pixel and second(right/down) pixel
	Vec3b hiddenPix1, hiddenPix2;
};

//recursive function used by outputUpdateGC to clear unwanted zone in graph cut result
void markAsOld(Mat& indicator, int x, int y) {
	if (x > 0 && indicator.at<uchar>(y, x - 1) == 127) {
		indicator.at<uchar>(y, x - 1) = 0;
		markAsOld(indicator, x - 1, y);
	}
	if (x < indicator.cols - 1 && indicator.at<uchar>(y, x + 1) == 127) {
		indicator.at<uchar>(y, x + 1) = 0;
		markAsOld(indicator, x + 1, y);
	}
	if (y > 0 && indicator.at<uchar>(y - 1, x) == 127) {
		indicator.at<uchar>(y - 1, x) = 0;
		markAsOld(indicator, x, y - 1);
	}
	if (y < indicator.rows - 1 && indicator.at<uchar>(y + 1, x) == 127) {
		indicator.at<uchar>(y + 1, x) = 0;
		markAsOld(indicator, x, y + 1);
	}
}

/*	update output and edge costs with graph cut on a specified rectangular grid of pixels
	parameters:
	const Mat&		input			input image (pattern)
	int				iPosX,iPosY		position(upper-left corner) of the overlapping zone for input image
	Mat&			output			output image
	int				oPosX,oPosY		position(upper-left corner) of the overlapping zone for output image
	Graph<..>&		gph				graph instance to be used for graph cut
	double*			hCuts,vCuts	costs for horizontal/vertical edges
	int				width,height	width/height of the overlapping zone
*/
void outputUpdateGC(const Mat &input, int iPosX, int iPosY, Mat &output, int oPosX, int oPosY, 
	Graph<double, double, double> &gph, Cut* hCuts, Cut* vCuts, int width, int height) {
	gph.reset();
	int outX = output.cols, outY = output.rows,
		inX = input.cols, inY = input.rows;
	//add all pixels in overlapping zone as vertices
	gph.add_node(width*height);
    
	//add edges for terminals source(old output) and sink(new patching)
    if (oPosX > 0) {
        for (int j = 0; j < height; j++) {
			gph.add_tweights(j*width, MAXVAL, 0.);
        }
    }
    if (oPosX + width <= outX) {
        for (int j = 0; j < height; j++) {
			gph.add_tweights((j + 1)*width - 1, MAXVAL, 0.);
        }
    }
    if (oPosY > 0) {
        for (int i = 0; i < width; i++) {
			gph.add_tweights(i, MAXVAL, 0.);
        }
    }
    if (oPosY + height <= outY) {
        for (int i = 0; i < width; i++) {
			gph.add_tweights((height - 1)*width + i, MAXVAL, 0.);
        }
    }
    
	//add edges between pixels
    for (int i = 0; i < width - 1; i++)	{	//horizontal edges
        for (int j = 0; j < height; j++) {
			if (hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost == -1.) {
				double c = cost(input, iPosX, iPosY, output, oPosX, oPosY, i, j, i + 1, j);
				gph.add_edge(i + j*width, i + j*width + 1, c, c);
			}
			else if (hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost == MAXVAL) {
				gph.add_tweights(i + j*width, 0., MAXVAL);
				gph.add_tweights(i + j*width + 1, 0., MAXVAL);
			} else {
				int id = gph.add_node();
				gph.add_tweights(id, 0., hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost);
				gph.add_edge(i + j*width, id, cost(
					output.at<Vec3b>(j + oPosY, i + oPosX), 
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix2,
					input.at<Vec3b>(j + iPosY, i + iPosX),
					input.at<Vec3b>(j + iPosY, i + 1 + iPosX)
					), MAXVAL);
				gph.add_edge(i + j*width + 1, id, cost(
					output.at<Vec3b>(j + oPosY, i + 1 + oPosX),
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix1,
					input.at<Vec3b>(j + iPosY, i + 1 + iPosX),
					input.at<Vec3b>(j + iPosY, i + iPosX)
					), MAXVAL);
			}
        }
    }
    for (int i = 0; i < width; i++)	{		//vertical edges
        for (int j = 0; j < height - 1; j++) {
			if (vCuts[i + oPosX + (j + oPosY)*outX].cost == -1.) {
				double c = cost(input, iPosX, iPosY, output, oPosX, oPosY, i, j, i, j + 1);
				gph.add_edge(i + j*width, i + (j + 1)*width, c, c);
			}
			else if (vCuts[i + oPosX + (j + oPosY)*outX].cost == MAXVAL) {
				gph.add_tweights(i + j*width, 0., MAXVAL);
				gph.add_tweights(i + (j + 1)*width, 0., MAXVAL);
			} else {
				int id = gph.add_node();
				gph.add_tweights(id, 0., vCuts[i + oPosX + (j + oPosY)*outX].cost);
				gph.add_edge(i + j*width, id, cost(
					output.at<Vec3b>(j + oPosY, i + oPosX),
					vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix2,
					input.at<Vec3b>(j + iPosY, i + iPosX),
					input.at<Vec3b>(j + 1 + iPosY, i + iPosX)
					), MAXVAL);
				gph.add_edge(i + (j + 1)*width, id, cost(
					output.at<Vec3b>(j + 1 + oPosY, i + oPosX),
					vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix1,
					input.at<Vec3b>(j + 1 + iPosY, i + iPosX),
					input.at<Vec3b>(j + iPosY, i + iPosX)
					), MAXVAL);
			}
        }
    }
    
	//graph cut!!!
	gph.maxflow();
    
/*
//	naive update taking all undefined zones as new parts (excessive part with artifacts)
//	delete all 2nd argument of function what_segment (", Graph<double, double, double>::SINK") to discard all undefined zones (holes in new patch) 
	//updatecuts
	for (int i = 0; i < width - 1; i++)		//horizontal edges
		for (int j = 0; j < height; j++){
			if (gph.what_segment(i + width*j, Graph<double, double, double>::SINK) == Graph<double, double, double>::SOURCE){
				if (gph.what_segment(i + width*j + 1, Graph<double, double, double>::SINK) == Graph<double, double, double>::SINK){
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost = cost(
						output.at<Vec3b>(j + oPosY, i + oPosX),
						hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix2,
						input.at<Vec3b>(j + iPosY, i + iPosX),
						input.at<Vec3b>(j + iPosY, i + 1 + iPosX)
						);
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix1 = input.at<Vec3b>(j + iPosY, i + iPosX);
				}
			}
			else{
				if (gph.what_segment(i + width*j + 1, Graph<double, double, double>::SINK) == Graph<double, double, double>::SINK){
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost = -1.;
				}
				else{
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost = cost(
						output.at<Vec3b>(j + oPosY, i + 1 + oPosX),
						hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix1,
						input.at<Vec3b>(j + iPosY, i + 1 + iPosX),
						input.at<Vec3b>(j + iPosY, i + iPosX)
						);
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix2 = input.at<Vec3b>(j + iPosY, i + 1 + iPosX);
				}
			}
		}
	for (int i = 0; i < width; i++)			//vertical edges
		for (int j = 0; j < height - 1; j++){
			if (gph.what_segment(i + width*j, Graph<double, double, double>::SINK) == Graph<double, double, double>::SOURCE){
				if (gph.what_segment(i + width*(j + 1), Graph<double, double, double>::SINK) == Graph<double, double, double>::SINK){
					vCuts[i + oPosX + (j + oPosY)*outX].cost = cost(
						output.at<Vec3b>(j + oPosY, i + oPosX),
						vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix2,
						input.at<Vec3b>(j + iPosY, i + iPosX),
						input.at<Vec3b>(j + 1 + iPosY, i + iPosX)
						);
					vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix1 = input.at<Vec3b>(j + iPosY, i + iPosX);
				}
			}
			else{
				if (gph.what_segment(i + width*(j + 1), Graph<double, double, double>::SINK) == Graph<double, double, double>::SINK){
					vCuts[i + oPosX + (j + oPosY)*outX].cost = -1.;
				}
				else{
					vCuts[i + oPosX + (j + oPosY)*outX].cost = cost(
						output.at<Vec3b>(j + 1 + oPosY, i + oPosX),
						vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix1,
						input.at<Vec3b>(j + 1 + iPosY, i + iPosX),
						input.at<Vec3b>(j + iPosY, i + iPosX)
						);
					vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix2 = input.at<Vec3b>(j + 1 + iPosY, i + iPosX);
				}
			}
		}
	//update output pixels
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++)
			if (gph.what_segment(i + width*j, Graph<double, double, double>::SINK) == Graph<double, double, double>::SINK)
				output.at<Vec3b>(j + oPosY, i + oPosX) = input.at<Vec3b>(j + iPosY, i + iPosX);
/*/
    
//	for consistent graphcut result
	//register node segmentation indication: 0-old 255-new 127-undefined
	Mat indicator(height, width, CV_8U);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (gph.what_segment(i + width*j, Graph<double, double, double>::SINK) == Graph<double, double, double>::SOURCE) {
				indicator.at<uchar>(j, i) = 0;
            } else if (gph.what_segment(i + width*j, Graph<double, double, double>::SOURCE) == Graph<double, double, double>::SINK) {
				indicator.at<uchar>(j, i) = 255;
            } else {
				indicator.at<uchar>(j, i) = 127;
            }
        }
    }
	//imshow("original cut result", indicator);
	//mark all undefined zones adjacent with old part as old
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (indicator.at<uchar>(j, i) == 0) {
				markAsOld(indicator, i, j);
            }
        }
    }
	//imshow("clearing of grey zone", indicator);
	//mark other undefined zones (inside new part) as new
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (indicator.at<uchar>(j, i) == 127) {
				indicator.at<uchar>(j, i) = 255;
            }
        }
    }
	//imshow("hole filled", indicator);
	//update cuts
    for (int i = 0; i < width - 1; i++)	{	//horizontal edges
		for (int j = 0; j < height; j++) {
			if (indicator.at<uchar>(j, i) == 0) {
				if (indicator.at<uchar>(j, i + 1) == 255) {
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost = cost(output.at<Vec3b>(j + oPosY, i + oPosX),hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix2,input.at<Vec3b>(j + iPosY, i + iPosX),input.at<Vec3b>(j + iPosY, i + 1 + iPosX));
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix1 = input.at<Vec3b>(j + iPosY, i + iPosX);
				}
			} else {
				if (indicator.at<uchar>(j, i + 1) == 255) {
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost = -1.;
				} else {
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].cost = cost(output.at<Vec3b>(j + oPosY, i + 1 + oPosX),hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix1,input.at<Vec3b>(j + iPosY, i + 1 + iPosX),input.at<Vec3b>(j + iPosY, i + iPosX));
					hCuts[i + oPosX + (j + oPosY)*(outX - 1)].hiddenPix2 = input.at<Vec3b>(j + iPosY, i + 1 + iPosX);
				}
			}
		}
    }
    for (int i = 0; i < width; i++) {			//vertical edges
		for (int j = 0; j < height - 1; j++) {
			if (indicator.at<uchar>(j, i) == 0) {
				if (indicator.at<uchar>(j + 1, i) == 255) {
					vCuts[i + oPosX + (j + oPosY)*outX].cost = cost(output.at<Vec3b>(j + oPosY, i + oPosX),vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix2,input.at<Vec3b>(j + iPosY, i + iPosX),input.at<Vec3b>(j + 1 + iPosY, i + iPosX));
					vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix1 = input.at<Vec3b>(j + iPosY, i + iPosX);
				}
            } else {
				if (indicator.at<uchar>(j + 1, i) == 255) {
					vCuts[i + oPosX + (j + oPosY)*outX].cost = -1.;
				}
				else {
					vCuts[i + oPosX + (j + oPosY)*outX].cost = cost(output.at<Vec3b>(j + 1 + oPosY, i + oPosX),vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix1,input.at<Vec3b>(j + 1 + iPosY, i + iPosX),input.at<Vec3b>(j + iPosY, i + iPosX));
					vCuts[i + oPosX + (j + oPosY)*outX].hiddenPix2 = input.at<Vec3b>(j + 1 + iPosY, i + iPosX);
				}
			}
		}
    }
	//update pixels
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            if (indicator.at<uchar>(j, i) == 255) {
				output.at<Vec3b>(j + oPosY, i + oPosX) = input.at<Vec3b>(j + iPosY, i + iPosX);
            }
        }
    }
}

//calculate position value
void calcPosition(int& i, int& o, int& l, int il, int ol) {
    cout << "AVANT : i: " << i << ". o: " << o << ". l: " << l << ". il: " << il << ". ol: " << ol << endl;
	if (o < 0) {
		i = -o;
		l = il + o;
		o = 0;
	} else if (o <= ol - il) {
		i = 0;
		l = il;
	} else {
		i = 0;
		l = ol - o;
	}
    cout << "APRES : i: " << i << ". o: " << o << ". l: " << l << ". il: " << il << ". ol: " << ol << endl;
}

//initial fill of output and cost arrays with repetition of input pattern
void initialFill(const Mat& input, Mat& output, Cut* hCuts, Cut* vCuts) {
	int outX = output.cols, outY = output.rows,
		inX = input.cols, inY = input.rows;
	//initial fill of output with repetition of input pattern
    for (int i = 0; i < outX; i++) {
        for (int j = 0; j < outY; j++) {
			output.at<Vec3b>(j, i) = input.at<Vec3b>(j%inY, i%inX);
        }
    }
	//update horizontal edge cost array
    for (int i = 0; i < outX - 1; i++) {
        for (int j = 0; j < outY; j++) {
            if (i%inX == inX - 1) {
				hCuts[i + j*(outX - 1)].cost = MAXVAL;
            } else {
				hCuts[i + j*(outX - 1)].cost = -1.;
            }
        }
    }
	//update vertical edge cost array
    for (int i = 0; i < outX; i++) {
        for (int j = 0; j < outY - 1; j++) {
            if (j%inY == inY - 1) {
				vCuts[i + j*outX].cost = MAXVAL;
            } else {
				vCuts[i + j*outX].cost = -1.;
            }
        }
    }
}

//draw cuts with specified thickness
void drawCuts(const Mat& output, const Cut* hCuts, const Cut* vCuts, Mat& disp, int thickness){
	int outX = output.cols, outY = output.rows;
	disp = output.clone();
    for (int i = 0; i < outX - 1; i++) {
		for (int j = 0; j < outY; j++) {
			if (hCuts[i + j*(outX - 1)].cost == MAXVAL) {
                for (int ii = max(i - thickness,0); ii < min(i + thickness + 2,outX); ii++) {
                    for (int jj = max(j - thickness,0); jj < min(j + thickness + 1,outY); jj++) {
						disp.at<Vec3b>(jj, ii) = Vec3b(255, 0, 255);
                    }
                }
			} else if (hCuts[i + j*(outX - 1)].cost > -1.) {
                for (int ii = max(i - thickness, 0); ii < min(i + thickness + 2, outX); ii++) {
                    for (int jj = max(j - thickness, 0); jj < min(j + thickness + 1, outY); jj++) {
						disp.at<Vec3b>(jj, ii) = Vec3b(255, 255, 0);
                    }
                }
			}
		}
    }
    for (int i = 0; i < outX; i++) {
		for (int j = 0; j < outY - 1; j++) {
			if (vCuts[i + j*outX].cost == MAXVAL) {
                for (int ii = max(i - thickness, 0); ii < min(i + thickness + 1, outX); ii++) {
                    for (int jj = max(j - thickness, 0); jj < min(j + thickness + 2, outY); jj++) {
						disp.at<Vec3b>(jj, ii) = Vec3b(255, 0, 255);
                    }
                }
			} else if (vCuts[i + j*outX].cost > -1.) {
                for (int ii = max(i - thickness, 0); ii < min(i + thickness + 1, outX); ii++) {
                    for (int jj = max(j - thickness, 0); jj < min(j + thickness + 2, outY); jj++) {
						disp.at<Vec3b>(jj, ii) = Vec3b(255, 255, 0);
                    }
                }
			}
		}
    }
}

/*	main function for texture generation
	parameters:
		String						inputTexturePath	path for input texture
		int							outX,outY			output width/height
		enum.PatchPlacementMode		mode				choice for patch placement methode
		bool						randTransform		enable/disable random transformation(rotation/symetry) for patch
		int							maxIter				maximum iteration of graph cut update
		int							pauseInterval		number of iterations before a regular pause for actual output display, <=0 to disable

	TODO:
		implementation
		transformation
		scaling?
*/
void textureGenerator(String inputTexturePath, int outX, int outY, PatchPlacementMode mode,
	bool randTransform, int maxIter, int pauseInterval) {
	//read and show input
	Mat input = imread(inputTexturePath);
	cout << "input type: " << (input.type()==CV_8UC3?"8 bit RGB":"not 8 bit RGB") << "\n";
	imshow("Input texture", input); waitKey();
	int inX = input.cols, inY = input.rows;
	//allocate output
	Mat output(outY, outX, input.type());
	//define and allocate graph
	Graph<double, double, double> gph(/*estimated # of nodes*/ inX*inY, /*estimated # of edges*/ 2 * inX*inY - inX - inY);
	//allocate 2 arrays to store information of former cuts, one for horizontal edges, the other for vertical ones
	Cut *hCuts = new Cut[(outX-1)*outY], *vCuts = new Cut[(outY-1)*outX];
	//initial fill of output and cost arrays with repetition of input pattern
	initialFill(input, output, hCuts, vCuts);
	//precompute 8 patch transformations if needed
	//TODO

	//iteration with designated patch placement methode until end requirement (maxiteration, low cost ...)
	switch(mode) {
        //	Random placement
        case RANDOM: {
            RandomPositionGenerator rpg(inX, inY, outX, outY);
            int iPosX, iPosY, oPosX, oPosY, width, height;
            Mat outDisp;
            imshow("initial output texture", output); waitKey();
            //destroyWindow("initial output texture");
            for (int i = 0; i < maxIter; i++) {
                rpg.changePosition(oPosX, oPosY);
                calcPosition(iPosX, oPosX, width, inX, outX);
                calcPosition(iPosY, oPosY, height, inY, outY);
                outputUpdateGC(input, iPosX, iPosY, output, oPosX, oPosY, gph, hCuts, vCuts, width, height);
                if (pauseInterval > 0 && i%pauseInterval == pauseInterval - 1) {
                    cout << "iteration: " << (i+1) << "\n";
                    drawCuts(output, hCuts, vCuts, outDisp, 0);
                    imshow("actual output texture", outDisp); waitKey();
                    imshow("actual output texture", output); waitKey();
                }
            }
            break;
        }
            
        //	Entire patch matching
        case ALLMATCH: {
            cout << "Entire patch matching\n";
            Mat mask = Mat::zeros(outY, outX, CV_8UC1);
            AllMatchPositionGenerator ampg(input, output, mask, 0.01);
            int iPosX, iPosY, oPosX, oPosY, width, height;
            Mat outDisp;
            imshow("initial output texture", output); waitKey();
            width = inX;
            height = inY;
            iPosX = 0;
            iPosY = 0;

            for (int i = 0; i < maxIter; i++) {
                ampg.change_position(oPosX, oPosY);
                calcPosition(iPosX, oPosX, width, inX, outX);
                calcPosition(iPosY, oPosY, height, inY, outY);
                cout << "width : " << width << endl;
                cout << "height : " << height << endl;
                outputUpdateGC(input, iPosX, iPosY, output, oPosX, oPosY, gph, hCuts, vCuts, width, height);
                // Update mask
                for (int k = 0; k < width; k++) {
                    for (int j = 0; j < height; j++) {
                        mask.at<uchar>(oPosY + j, oPosX + k) = 255;
                    }
                }
                imshow("mask", mask); waitKey();
                if (pauseInterval > 0 && i%pauseInterval == pauseInterval - 1) {
                    cout << "iteration: " << (i+1) << "\n";
                    drawCuts(output, hCuts, vCuts, outDisp, 0);
                    imshow("actual output texture", outDisp); waitKey();
                    imshow("actual output texture", output); waitKey();
                }
            }
            break;
        }
            
        //	Sub-patch matching
        case SUBPATCH:
            //TODO
            cout << "Sub-patch matching\n";
            break;
	}
    
	//display output
	destroyWindow("actual output texture");
	imshow("Final output texture", output); waitKey();
	//delete dynamic array
	delete[] hCuts, vCuts;
}

int main(int argc, const char * argv[]) {

    /*
     * Random Position Generator test/tutorial
     */
/*
	//construct an rpg instance for a 8*10 patch and 16*15 output (size in X*Y) 
	RandomPositionGenerator rpg(8, 10, 16, 15);
	//declare two variables for position storage
	int posx, posy;
	//print 50 randomly-generated positions
	for (int i = 0; i < 50; i++) {
		rpg.changePosition(posx, posy);
		cout << posx << ", " << posy << "\n";
	}
 */
    
    /*
     * All Match Position Generator test/tutorial
     */
    Mat input = imread("../../grass2.jpg");
    Mat output = Mat::zeros(480, 640, input.type());
    Mat mask = Mat::zeros(output.rows, output.cols, CV_8U);
    for (int i = 0; i < input.cols; i++) {
        for (int j = 0; j < input.rows; j++) {
            output.at<Vec3b>(output.rows/2 - input.rows/2 + j, output.cols/2 - input.cols/2 + i) = input.at<Vec3b>(j, i);
            mask.at<uchar>(output.rows/2 - input.rows/2 + j, output.cols/2 - input.cols/2 + i) = 255;
        }
    }

//	textureGenerator("../../strawberries.jpg", 640, 480, RANDOM, false, 200, 10);
	//textureGenerator("../../grass.jpg", 640, 480, RANDOM, false, 200, 10);
//	textureGenerator("../../grass2.jpg", 640, 480, ALLMATCH, false, 200, 10);
	textureGenerator("../../bark.tiff", 320, 240, RANDOM, false, 200, 10);

    return 0;
}
