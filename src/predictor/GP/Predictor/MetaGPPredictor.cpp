/*

 * MetaGPPredictor.cpp
 *
 *  Created on: Apr 7, 2014
 *      Author: zhoujingbo


#include "MetaGPPredictor.h"
#include "../gp/optimisation/SCGModelTrainer.h"

MetaGPPredictor::MetaGPPredictor()
{
	// TODO Auto-generated constructor stub

}

MetaGPPredictor::~MetaGPPredictor()
{
	// TODO Auto-generated destructor stub

	clearPrdMat();//clear the matrix of predictor instance
}


void MetaGPPredictor<T>::clearPrdMat() {
	for (int i = 0; i < numL; i++) {
		for (int j = 0; j < numK; j++) {
			if (NULL != PrdMat[i][j]) {
				delete PrdMat[i][j];
			}
			PrdMat[i][j]=NULL;
		}
	}
	PrdMat.clear();
}

void MetaGPPredictor<T>::initPrdMat() {

	PrdMat.resize(numL);
	for (int i = 0; i < numL; i++) {
		PrdMat[i].resize(numK);
		for (int j = 0; j < numK; j++) {
			PrdMat[i][j] = NULL;
		}
	}

}



void MetaGPPredictor<T>::initCovFuncMat(double range, double sill, double nugget){



	gaussianCFMat.resize(numL);
	nuggetCFMat.resize(numL);
	sumCFMat.resize(numL);

	for(int i=0;i<numL;i++){
		gaussianCFMat[i].resize(numK);
		nuggetCFMat[i].clear();
		sumCFMat[i].resize(numK);

		for(int j=0; j<numK; j++){
			gaussianCFMat[i][j] = new GaussianCF(range,sill);
			nuggetCFMat[i].push_back(WhiteNoiseCF(nugget));
			sumCFMat[i][j].addCovarianceFunction(*(gaussianCFMat[i][j]));
			sumCFMat[i][j].addCovarianceFunction(nuggetCFMat[i][j]);
		}
	}
}

void MetaGPPredictor<T>::setDefaultCovFuncPara(double range, double sill, double nugget){
	this->range = range;
	this->sill = sill;
	this->nugget = nugget;
}

void MetaGPPredictor<T>::clearCovFuncMat() {

	nuggetCFMat.clear();
	sumCFMat.clear();

	for (int i = 0; i < numL; i++) {
		for (int j = 0; j < numK; j++) {
			if (NULL != gaussianCFMat[i][j]) {
				delete gaussianCFMat[i][j];
			}
			gaussianCFMat[i][j] = NULL;
		}
	}
	gaussianCFMat.clear();
}








*/
