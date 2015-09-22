/*
 * MetaGPPredictor.h
 *
 *  Created on: Apr 7, 2014
 *      Author: zhoujingbo
 */

#ifndef METAGPPREDICTOR_H_
#define METAGPPREDICTOR_H_


/**
*     inherit graph: Predictor=>MetaGPPredictor=>LOOCVPredictor (i.e. LOOCVDirPredictor) =>LOOCVItrPredictor
 *     					  \\
 *     						== RegPredictor
**/


#include "Predictor.h"
#include "../gp/gaussianProcesses/MetaGP.h"

#include "../gp/covarianceFunctions/SumCovarianceFunction.h"//add by jignbo
#include "../gp/covarianceFunctions/GaussianCF.h"
#include "../gp/covarianceFunctions/WhiteNoiseCF.h"
#include "../gp/covarianceFunctions/ARDGaussianCF.h"
#include "../gp/covarianceFunctions/CovarianceFunction.h"

template <class T>
class MetaGPPredictor: public Predictor<T> {
public:

	MetaGPPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<T> & dataXtst,
			double range, double sill, double nugget) :
			Predictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst) {

		initPrdMat(); //initialize the matrix of predictor instances
		setDefaultCovFuncPara(range, sill, nugget);
		initCovFuncMat(range, sill, nugget);

	}

	//constructor for multiple queries, only create the predictor with the common configuration
	MetaGPPredictor(vector<int>& Lvec, vector<int>& Kvec, double range,
			double sill, double nugget) :
			Predictor<T>(Lvec, Kvec) {

		initPrdMat(); //initialize the matrix of predictor instances
		setDefaultCovFuncPara(range, sill, nugget);
		initCovFuncMat(range, sill, nugget);

	}

	virtual ~MetaGPPredictor() {
		clearCovFuncMat();
		clearPrdMat(); //clear the matrix of predictor instance
	}

	void setDefaultCovFuncPara(double range, double sill, double nugget) {
			this->range = range;
			this->sill = sill;
			this->nugget = nugget;
	}

protected:
	//================================= abstract interface of Prediction part ===========================
	virtual void inferPrediction()=0;
	virtual void inferItrPrediction() = 0;


	//for direct prediction, nothing to do with Xtrn
	//for iterative prediction, call "shiftConcatXtrn()"
	//implement for different prediction functions
	virtual void updateNextXtrn()=0;
	//

protected:
	vector<vector<MetaGP*> > PrdMat; //two dimension matrix with different k and l
	//matrix of covFunc for PrdMat
	vector<vector<CovarianceFunction *> > gaussianCFMat;
	vector<vector<WhiteNoiseCF*> > nuggetCFMat;
	vector<vector<SumCovarianceFunction> > sumCFMat;
	double range, sill, nugget;	//initial value of covFunc


private:


	void initPrdMat() {

		PrdMat.resize(this->numL);
		for (int i = 0; i < this->numL; i++) {
			PrdMat[i].resize(this->numK);
			for (int j = 0; j < this->numK; j++) {
				PrdMat[i][j] = NULL;
			}
		}

	}

	void clearPrdMat(){
		for (int i = 0; i < this->numL; i++) {
			for (int j = 0; j < this->numK; j++) {
				if (NULL != PrdMat[i][j]) {
					delete PrdMat[i][j];
				}
				PrdMat[i][j] = NULL;
			}
		}
		PrdMat.clear();
	}

	/**
	 *
	 */
	void initCovFuncMat(double range, double sill, double nugget) {

		gaussianCFMat.resize(this->numL);
		nuggetCFMat.resize(this->numL);
		sumCFMat.resize(this->numL);

		for (int i = 0; i < this->numL; i++) {
			gaussianCFMat[i].resize(this->numK);
			nuggetCFMat[i].resize(this->numK);
			sumCFMat[i].resize(this->numK);

			for (int j = 0; j < this->numK; j++) {
				gaussianCFMat[i][j] = new GaussianCF(range, sill);//for exp
				//gaussianCFMat[i][j] = new ARDGaussianCF(range, sill, this->Lvec[i]);//for exp
				nuggetCFMat[i][j] = new WhiteNoiseCF(nugget);
				sumCFMat[i][j].addCovarianceFunction(*(gaussianCFMat[i][j]));
				sumCFMat[i][j].addCovarianceFunction(*(nuggetCFMat[i][j]));
			}
		}
	}

	void clearCovFuncMat() {


		for (int i = 0; i < this->numL; i++) {
			for (int j = 0; j < this->numK; j++) {
				if (NULL != gaussianCFMat[i][j]) {
					delete gaussianCFMat[i][j];
					gaussianCFMat[i][j] = NULL;
				}
				if(NULL != nuggetCFMat[i][j]){
					delete nuggetCFMat[i][j];
					nuggetCFMat[i][j] = NULL;
				}

			}
		}

		gaussianCFMat.clear();
		nuggetCFMat.clear();

		sumCFMat.clear();


	}



};

#endif /* METAGPPREDICTOR_H_ */
