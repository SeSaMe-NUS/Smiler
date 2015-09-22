/*
 * RegPredictor.h
 *
 *  Created on: Apr 3, 2014
 *      Author: zhoujingbo
 */

#ifndef REGPREDICTOR_H_
#define REGPREDICTOR_H_

#include <vector>
using namespace std;

#include "Predictor.h"
#include "../gp/gaussianProcesses/KnnReg.h"

template<class T>
class RegPredictor: public Predictor<T> {
public:
	//RegPredictor(vector<int> _Lvec, vector<int> _Kvec);
	RegPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<T> & dataXtst) :
			Predictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst) {

		initPrdMat();

	}

	//RegPredictor(vector<int> _Lvec, vector<int> _Kvec);
	RegPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<vector<T> >& weight,
			const vector<T> & dataXtst) :
			Predictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst) {

		initPrdMat();
		this->setWeightVec(weight);

	}

	//constructor for multiple queries, only create the predictor with the common configuration
	RegPredictor(vector<int>& Lvec, vector<int>& Kvec) :
		Predictor<T>(Lvec, Kvec) {
		initPrdMat();
	}


	virtual ~RegPredictor() {
		clearPrdMat();
	}

private:
	virtual void inferPrediction(){

		for (int i = 0; i <  Predictor<T>::numL; i++) {

			mat& XtrnSlice = this->Xtrn[i];
			vec& YtrnSlice = this->Ytrn[i];
			vec& weightSlice = this->weightVec[i];

			for (int j = 0; j < this->numK; j++) {
				mat XtrnSegm = XtrnSlice.rows(0, this->Kvec[j] - 1);
				vec YtrnSegm = YtrnSlice.rows(0, this->Kvec[j] - 1);
				vec weightSgm = weightSlice.rows(0, this->Kvec[j] - 1);

				//noted:
				if (NULL == this->PrdMat[i][j] ) {
					this->PrdMat[i][j] = new KnnReg();
				}

				// Compute the posterior GP and store the result in mean and var
				vec meanStep = vec().zeros(1);
				vec varStep = vec().zeros(1);
				PrdMat[i][j]->makePredictions(YtrnSegm, weightSgm, meanStep, varStep);// for exp
				//PrdMat[i][j]->makePredictions(YtrnSegm, meanStep, varStep);// for exp


				assert(meanStep.n_rows == 1 && varStep.n_rows == 1);
				this->mean[i][j].insert_rows(this->iterativeSteps, meanStep);
				this->var[i][j].insert_rows(this->iterativeSteps, varStep);

			}
		}
	//	this->iterativeSteps++;
	}

	virtual void inferItrPrediction() {
		inferPrediction();

		this->iterativeSteps++;
	}


	virtual void updateNextXtrn(){
		//for direct prediction, nothing to do with Xtrn
		return;
	}

	/**
	 * make next prediction
	 */
	/*void inferItrPrediction(const vector<vector<T> >& dataYtrn) {
		this->shiftConcatXtrn();
		updateYtrn(dataYtrn);
		updateXtst();
		clearPrdMat();
		inferPrediction();
	}*/







private:
	vector<vector<KnnReg*> > PrdMat; //two dimension matrix with different k and l

private:

	void clearPrdMat() {
		for (int i = 0; i < this->numL; i++) {
			for (int j = 0; j < this->numK; j++) {
				if (NULL != PrdMat[i][j]) {
					delete PrdMat[i][j];
				}
				PrdMat[i][j] = NULL;

			}
			//PrdMat[i].clear();
		}
		//PrdMat.clear();
	}

	void initPrdMat() {

		PrdMat.resize(this->numL);
		for (int i = 0; i < this->numL; i++) {
			PrdMat[i].resize(this->numK);
			for (int j = 0; j < this->numK; j++) {
				PrdMat[i][j] = NULL;
			}
		}

	}

	void updateXtst() {
		this->Xtst.shed_col(0);
		vec tstVec(this->Xtst.n_rows);

		double meanPre, varPre;
		this->getPredictonResult(meanPre, varPre);
		tstVec.fill(meanPre);
		this->Xtst.insert_cols(this->Xtst.n_cols, tstVec);

	}

};

#endif /* REGPREDICTOR_H_ */
