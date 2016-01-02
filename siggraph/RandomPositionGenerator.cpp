#include <random>
#include "RandomPositionGenerator.h" 

RandomPositionGenerator::RandomPositionGenerator(const int patchX, const int patchY, const int outputX, const int outputY){
    std::random_device rd;
	dre = std::default_random_engine(rd());
	rndX = std::uniform_int_distribution<int>(1 - patchX, outputX - 1);
	rndY = std::uniform_int_distribution<int>(1 - patchY, outputY - 1);
}

void RandomPositionGenerator::changePosition(int& posX, int& posY){
	posX = rndX(dre);
	posY = rndY(dre);
}
