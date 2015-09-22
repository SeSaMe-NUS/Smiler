/*
 * LOOCVPredictor.h
 *
 *  Created on: Apr 7, 2014
 *      Author: zhoujingbo
 */

#ifndef LOOCVPREDICTOR_H_
#define LOOCVPREDICTOR_H_

/**
*     inherit graph: Predictor=>MetaGPPredictor=>LOOCVPredictor (i.e. LOOCVDirPredictor) =>LOOCVItrPredictor
 *     					  \\
 *     						== RegPredictor
**/


#include "MetaGPPredictor.h"
#include "../gp/gaussianProcesses/LOOCVGP.h"
#include "../gp/gaussianProcesses/MMLGP.h"
#include "../../../tools/BladeLoader.h"
#include "../gp/optimisation/SCGModelTrainer.h"
#include "UtlPredictor.h"

template <class T>
class LOOCVPredictor: public MetaGPPredictor<T> {
public:
	LOOCVPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<T> & dataXtst,
			double range, double sill, double nugget) :
			MetaGPPredictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst, range,
					sill, nugget) {


	}

	LOOCVPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<T> & dataXtst,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<vector<T> >& weight,
			double range, double sill, double nugget) :
			MetaGPPredictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst, range,
					sill, nugget) {
		this->setWeightVec(weight);
	}

	//constructor for multiple queries, only create the predictor with the common configuration
	LOOCVPredictor(vector<int>& Lvec, vector<int>& Kvec, double range,
			double sill, double nugget) :
			MetaGPPredictor<T>(Lvec, Kvec, range, sill, nugget) {

	}

	virtual ~LOOCVPredictor() {
	}

//private:
//	vector<vector<LOOCVGP*> > PrdMat; //two dimension matrix with different k and l


public:
	void makeCntPrediction(){

	}

private:


	virtual void inferPrediction() {

		//cout << "iterativeSteps:" <<this-> iterativeSteps << endl;
		for (int i = 0; i < this->numL; i++) {

			mat& XtrnSlice = this->Xtrn[i];
			vec& YtrnSlice = this->Ytrn[i];
			vec& weightSlice = this->weightVec[i];
			//select the tailed (i.e. latest) L elements as test data
			mat XtstSlice = this->Xtst.cols(this->Xtst.n_cols - this->Lvec[i], this->Xtst.n_cols - 1); //only record the Xtst with the maximum length (maximum dimensions)
			int dim = this->Lvec[i];

			for (int j = 0; j < this->numK; j++) {
				mat XtrnSgm = XtrnSlice.rows(0, this->Kvec[j] - 1);
				vec YtrnSgm = YtrnSlice.rows(0, this->Kvec[j] - 1);
				vec weightSgm = weightSlice.rows(0, this->Kvec[j] - 1);

				vec meanStep = vec().zeros(1);
				vec varStep = vec().ones(1);

				if (NULL == this->PrdMat[i][j]) {

					this->PrdMat[i][j] = new LOOCVGP(dim, 1, XtrnSgm, YtrnSgm,weightSgm,
							this->sumCFMat[i][j]);
				}

//				this->PrdMat[i][j] = inferPredictionSgm(XtrnSgm, YtrnSgm, weightSgm,
//						XtstSlice, dim, this->sumCFMat[i][j], *(this->gaussianCFMat[i][j]),
//						meanStep, varStep);
				if(this->weightPrdMat[i][j]>=this->weightPrdMat_cutoff){//make predicton if weight is larger than 0

				inferPredictionSgm(this->PrdMat[i][j],
								 XtrnSgm,  YtrnSgm, weightSgm,
								 XtstSlice,
								 this->sumCFMat[i][j], meanStep, varStep);
				}

				assert(meanStep.n_rows == 1 && varStep.n_rows == 1);

				this->mean[i][j].insert_rows(this->iterativeSteps, meanStep);
				this->var[i][j].insert_rows(this->iterativeSteps, varStep);

			}
		}

		//this->iterativeSteps++;
	}

	/**
	 * make prediction based on one segement of the data,
	 *  i.e. corresponding to one (l,k) element in predictor matrix
	 */
//	LOOCVGP* inferPredictionSgm(mat& XtrnSgm, vec& YtrnSgm, vec & weightSgm,
//			mat& XtstSlice, int dim, CovarianceFunction& _sumCF,
//			CovarianceFunction& _gaussianCF, vec & _meanSgm, vec& _varSgm) {
//
//		LOOCVGP* loocv = new LOOCVGP(dim, 1, XtrnSgm, YtrnSgm, _sumCF,
//				weightSgm);
//		SCGModelTrainer scggp(*loocv);
//		scggp.setDisplay(this->display);
//		scggp.Train(UtlPredictor_namespace::trainStep);
//		//scggp.Train(25);
//		//loocv->printCovPara(_sumCF);
//		loocv->makePredictions(_meanSgm, _varSgm, XtstSlice, _gaussianCF);
//
//		return loocv;
//	}


	/**
	 * make prediction based on one segement of the data,
	 *  i.e. corresponding to one (l,k) element in predictor matrix
	 */
	void inferPredictionSgm(MetaGP* metaGP,
				mat& XtrnSgm, vec& YtrnSgm, vec& weightSgm,
				mat& XtstSlice,
				CovarianceFunction& _gaussianCF, vec & _meanSgm, vec& _varSgm){

			metaGP->setXYtrn(XtrnSgm, YtrnSgm);
			metaGP->setWeight(weightSgm);
			SCGModelTrainer scggp(*metaGP);
			scggp.setDisplay(this->display);


			//for exp
			vec para = zeros(3);
			para = _gaussianCF.getParameters();
//			if(para(2)<= (-0.00109249+0.000001)&&(para(2)>=(-0.00109249-0.000001))){
			para(2) = 1;//for stability, avoid the singular matrix
//			}
//
			_gaussianCF.setParameters(para);
			//end for exp
			//metaGP->printCovPara(_gaussianCF);//
			scggp.Train(UtlPredictor_namespace::trainStep);//for exp

			metaGP->makePredictions(_meanSgm, _varSgm, XtstSlice, _gaussianCF);

		}




	/**
	 * make perdiction based on the Xtrn and Ytrn, the Xtrn and Ytrn should have been configured.
	 * This is to make direct prediction without error propagation for iterative prediction
	 */
	virtual void inferItrPrediction() {
		inferPrediction();
		this->iterativeSteps++;
	}



	//for direct prediction, nothing to do with Xtrn
	//for iterative prediction, call "shiftConcatXtrn()"
	//implement for different prediction functions
	virtual void updateNextXtrn() {

		//this->shiftConcatXtrn();
		return;
	}


};

#endif /* LOOCVPREDICTOR_H_ */
