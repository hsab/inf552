//
//  AllMatchPositionGenerator.hpp
//  siggraph
//
//  Created by Camille MASSET on 08/01/2016.
//
//

#ifndef AllMatchPositionGenerator_hpp
#define AllMatchPositionGenerator_hpp

#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//generate entire patch matching position for any given sizes of patch and output
//part of the patch can be out of the output zone
class AllMatchPositionGenerator {
private:
    double k;
    double sigma;
    cv::Mat input;
    cv::Mat output;
    cv::Mat mask; // to check if a pixel of output has already been synthetized : 255 = yes, 0 = no
    cv::Mat costs;
    cv::Mat probas;
    std::default_random_engine dre;
    std::uniform_int_distribution<int> rndX,rndY;
    std::discrete_distribution<> distrib;
    void compute_cost(cv::Point t);
    void compute_costs();
    int setup_generator();
    
public:
    //constructor
    AllMatchPositionGenerator(cv::Mat& input, cv::Mat& output, cv::Mat& mask, double k);
    
    //change to next translation
    void change_position(int& posX, int& posY);
};

#endif /* AllMatchPositionGenerator_hpp */
