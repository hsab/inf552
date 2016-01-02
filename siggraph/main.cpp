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

#include "RandomPositionGenerator.h"
#include "maxflow/graph.h"


using namespace std;
using namespace cv;

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

int main(int argc, const char * argv[]) {

//* Random Position Generator test/tutorial
	//construct an rpg instance for a 8*10 patch and 16*15 output (size in X*Y) 
	RandomPositionGenerator rpg(8, 10, 16, 15);
	//declare two variables for position storage
	int posx, posy;
	//print 50 randomly-generated positions
	for (int i = 0; i < 50; i++){
		rpg.changePosition(posx, posy);
		std::cout << posx << ", " << posy << "\n";
	}
//*/

    // Récupérer l'input
    Mat input = imread("../../strawberries.jpg");
    imshow("Input", input); waitKey();
    
    // Déterminer la taille de l'output
    Mat output = Mat(480, 640, CV_32FC1);
    // Tant que l'output n'est pas complet
        // Positionner un premier patch
        // Découper le patch par graphcut
    
    std::cout << "Hello, World!\n";
    return 0;
}
