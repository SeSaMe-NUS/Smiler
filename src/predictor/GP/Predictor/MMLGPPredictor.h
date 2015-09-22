/*
 * MMLGPPredictor.h
 *
 *  Created on: Aug 19, 2014
 *      Author: zhoujingbo
 */

#ifndef MMLGPPREDICTOR_H_
#define MMLGPPREDICTOR_H_

/**
*     inherit graph: Predictor=>MetaGPPredictor=>LOOCVPredictor (i.e. LOOCVDirPredictor) =>LOOCVItrPredictor
 *     					  \\
 *     						== RegPredictor
**/


#include "MetaGPPredictor.h"
#include "../gp/gaussianProcesses/MMLGP.h"
#include "../../../tools/BladeLoader.h"
#include "../gp/optimisation/SCGModelTrainer.h"
#include "UtlPredictor.h"

template<class T>
class MMLGPPredictor: public MetaGPPredictor<T> {
public:

	MMLGPPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<T> & dataXtst,
			double range, double sill, double nugget) :
			MetaGPPredictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst, range,
					sill, nugget) {

	}

	//constructor for multiple queries, only create the predictor with the common configuration
	MMLGPPredictor(vector<int>& Lvec, vector<int>& Kvec, double range,
			double sill, double nugget) :
			MetaGPPredictor<T>(Lvec, Kvec, range, sill, nugget) {

	}

	virtual ~MMLGPPredictor() {
	}

private:

	virtual void inferPrediction() {

		//cout << "iterativeSteps:" <<this-> iterativeSteps << endl;
		for (int i = 0; i < this->numL; i++) {

			mat& XtrnSlice = this->Xtrn[i];
			vec& YtrnSlice = this->Ytrn[i];
			vec& weightSlice = this->weightVec[i];
			//select the tailed (i.e. latest) L elements as test data
			mat XtstSlice = this->Xtst.cols(this->Xtst.n_cols - this->Lvec[i],
					this->Xtst.n_cols - 1); //only record the Xtst with the maximum length (maximum dimensions)
			int dim = this->Lvec[i];

			for (int j = 0; j < this->numK; j++) {
				mat XtrnSgm = XtrnSlice.rows(0, this->Kvec[j] - 1);
				vec YtrnSgm = YtrnSlice.rows(0, this->Kvec[j] - 1);
				vec weightSgm = weightSlice.rows(0, this->Kvec[j] - 1);

				vec meanStep = vec().zeros(1);
				vec varStep = vec().zeros(1);

				if (NULL == this->PrdMat[i][j]) {
					this->PrdMat[i][j] = new MMLGP(dim, 1, XtrnSgm, YtrnSgm,
							this->sumCFMat[i][j]);
				}

//				this->PrdMat[i][j] = inferPredictionSgm(XtrnSgm, YtrnSgm, weightSgm,
//						XtstSlice, dim, this->sumCFMat[i][j], *(this->gaussianCFMat[i][j]),
//						meanStep, varStep);

				inferPredictionSgm(this->PrdMat[i][j], XtrnSgm, YtrnSgm,
						XtstSlice, this->sumCFMat[i][j], meanStep, varStep);

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
	void inferPredictionSgm(MetaGP* metaGP, mat& XtrnSgm, vec& YtrnSgm,
			mat& XtstSlice, CovarianceFunction& _gaussianCF, vec & _meanSgm,
			vec& _varSgm) {

		metaGP->setXYtrn(XtrnSgm, YtrnSgm);
		SCGModelTrainer scggp(*metaGP);
		scggp.setDisplay(this->display);
		scggp.Train(UtlPredictor_namespace::trainStep);//for exp
		metaGP->printCovPara(_gaussianCF);

		metaGP->makePredictions(_meanSgm, _varSgm, XtstSlice, _gaussianCF);

	}

	/**
	 * make perdiction based on the Xtrn and Ytrn,
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

#endif /* MMLGPPREDICTOR_H_ */
