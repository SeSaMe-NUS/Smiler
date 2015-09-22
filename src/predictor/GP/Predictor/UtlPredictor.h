/*
 * UtlPredictor.h
 *
 *  Created on: Apr 7, 2014
 *      Author: zhoujingbo
 */

#ifndef UTLPREDICTOR_H_
#define UTLPREDICTOR_H_


#include <vector>
#include <cmath>
using namespace std;

//for arma
#include "armadillo"
using namespace arma;

class UtlPredictor{

public:
	//range, double sill, double nugget


};


namespace UtlPredictor_namespace{
void printMatrix(mat data);
void printVector(vec data);

static int trainStep = 5;//
static double minNugget = 0.01;

static double weightPrdMat_learnRate= 0.5;
static double weightPrdMat_cutoffRate = 0.5;//or 0.95
static int weightPrdMat_minRecoverStep = 2;
static int weightPrdMat_baseRecoverStep = 2;
static int weightPrdmat_maxRecoverStep_exp = 5;
static int weightPrdMat_maxRecoverStep = (int)pow(weightPrdMat_baseRecoverStep,weightPrdmat_maxRecoverStep_exp)*weightPrdMat_minRecoverStep;
}


/*

class UtlPredictor {
public:
	UtlPredictor();
	virtual ~UtlPredictor();

public:


};
*/

#endif /* UTLPREDICTOR_H_ */
