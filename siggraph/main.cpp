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

#include "maxflow/graph.h"

using namespace std;
using namespace cv;

int main(int argc, const char * argv[]) {
    // Récupérer l'input
    Mat input = imread("../strawberries.jpg");
    imshow("Input", input); waitKey();
    
    // Déterminer la taille de l'output
    // Tant que l'output n'est pas complet
        // Positionner un premier patch
        // Découper le patch par graphcut
    
    std::cout << "Hello, World!\n";
    return 0;
}
