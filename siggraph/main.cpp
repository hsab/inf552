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

#include "RandomPositionGenerator.h"
#include "maxflow/graph.h"

using namespace std;
using namespace cv;

//enumeration for patch placement mathode selection
enum PatchPlacementMode{RANDOM, ALLMATCH, SUBMATCH};
//maximum possible value for graph cut
const double MAXVAL = numeric_limits<double>::has_infinity ? numeric_limits<double>::infinity() : numeric_limits<double>::max();

float cost(Point s, Point t, const Mat& A, const Mat& B) {
    return norm(A.at<float>(s) - B.at<float>(s)) + norm(A.at<float>(t) - B.at<float>(t));
}

/* Buggy for immense output size (output-patch > RAND_MAX), non-uniform distribution
//get patch and output size, generate random patch position (upper left corner pixel position)
inline void getRandomPosition(const int patchX, const int patchY, const int outputX, const int outputY, int& posX, int& posY){
	posX = rand() % (outputX - patchX + 1);
	posY = rand() % (outputY - patchY + 1);
}
*/

/*	update output and edge costs with graph cut on a specified rectangular grid of pixels
	parameters:
	const Mat&		input			input image (pattern)
	int				iPosX,iPosY		position(upper-left corner) of the overlapping zone for input image
	Mat&			output			output image
	int				oPosX,oPosY		position(upper-left corner) of the overlapping zone for output image
	Graph<..>&		gph				graph instance to be used for graph cut
	double*			hCosts,vCosts	costs for horizontal/vertical edges
	int				width,height	width/height of the overlapping zone
*/
void outputUpdateGC(const Mat &input, int iPosX, int iPosY, Mat &output, int oPosX, int oPosY, 
	Graph<double, double, double> &gph, double* hCosts, double* vCosts, int width, int height){
	gph.reset();
	//TODO

}

//calculate position value
inline void calcPosition(int& i, int& o, int& l,int il, int ol){
	if (o < 0){
		i = -o;
		l = il + o;
		o = 0;
	}
	else if (o <= ol - il){
		i = 0;
		l = il;
	}
	else{
		i = 0;
		l = ol - o;
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
		add maxiteration argument and/or other end condition
		add argument to pause everytime for a designated number of iterartions and show result
		scaling?
*/
void textureGenerator(String inputTexturePath,int outX,int outY, PatchPlacementMode mode, 
	bool randTransform, int maxIter, int pauseInterval){
	//read and show input
	Mat input = imread(inputTexturePath);
	cout << "input type: " << (input.type()==CV_8UC3?"8 bit RGB":"not 8 bit RGB") << "\n";
	imshow("Input texture", input);
	waitKey();
	int inX = input.cols, inY = input.rows;
	//allocate output
	Mat output(outY,outX,input.type());
	//define and allocate graph
	Graph<double, double, double> gph(/*estimated # of nodes*/ inX*inY, /*estimated # of edges*/ 2 * inX*inY - inX - inY);
	//allocate 2 arrays to store information of former cuts, one for horizontal edges, the other for vertical ones
	double *hCosts = new double[(outX-1)*outY], *vCosts = new double[(outY-1)*outX];
	//initial fill of output with repetition of input pattern
	for (int i = 0; i < outX; i++)
		for (int j = 0; j < outY; j++)
			output.at<Vec3b>(j, i) = input.at<Vec3b>(j%inY,i%inX);
	//precompute 8 patch transformations if needed
	//TODO

	//iteration with designated patch placement methode until end requirement (maxiteration, low cost ...)
	switch(mode){
	//	Random placement
	case RANDOM:{
		RandomPositionGenerator rpg(inX, inY, outX, outY);
		int iPosX, iPosY, oPosX, oPosY, width, height;
		for (int i = 0; i < maxIter; i++){
			rpg.changePosition(oPosX, oPosY);
			calcPosition(iPosX, oPosX, width, inX, outX);
			calcPosition(iPosY, oPosY, height, inY, outY);
			outputUpdateGC(input, iPosX, iPosY, output, oPosX, oPosY, gph, hCosts, vCosts, width, height);
			if (pauseInterval > 0 && i%pauseInterval == pauseInterval - 1){
				imshow("actual output texture", output);
				cout << "iteration: " << (i+1) << "\n";
				waitKey();
				//destroyWindow("actual output texture");
			}
		}
	}
		break;
	//	Entire patch matching
	case ALLMATCH:
		//TODO
		cout << "Entire patch matching\n";
		break;
	//	Sub-patch matching
	case SUBMATCH:
		//TODO
		cout << "Sub-patch matching\n";
		break;
	}
	//display output
	destroyWindow("actual output texture");
	imshow("Final output texture", output); 
	waitKey();
	//delete dynamic array
	delete[] hCosts,vCosts;
}

int main(int argc, const char * argv[]) {

//* Random Position Generator test/tutorial
	//construct an rpg instance for a 8*10 patch and 16*15 output (size in X*Y) 
	RandomPositionGenerator rpg(8, 10, 16, 15);
	//declare two variables for position storage
	int posx, posy;
	//print 50 randomly-generated positions
	for (int i = 0; i < 50; i++){
		rpg.changePosition(posx, posy);
		cout << posx << ", " << posy << "\n";
	}
//*/
/*
	// Récupérer l'input
    Mat input = imread("../../strawberries.jpg");
    imshow("Input", input); waitKey();
    
    // Déterminer la taille de l'output
    Mat output = Mat(480, 640, CV_32FC1);
    // Tant que l'output n'est pas complet
        // Positionner un premier patch
        // Découper le patch par graphcut
//*/
	textureGenerator("../../strawberries.jpg", 640, 480, RANDOM, false, 100, 10);
    return 0;
}
