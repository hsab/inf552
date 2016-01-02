#ifndef SYTRUS_RANDOMPOSITIONGENERATOR_H
#define SYTRUS_RANDOMPOSITIONGENERATOR_H

#include <random>

//generate Random position for any given sizes of patch and output 
//part of the patch can be out out the output zone
class RandomPositionGenerator{
	
	private:
		std::default_random_engine dre;
		std::uniform_int_distribution<int> rndX,rndY;
	
	public:
		//constructor
		RandomPositionGenerator(const int patchX, const int patchY, const int outputX, const int outputY);

		//change to next random position
		void changePosition(int& posX, int& posY);
};

#endif
