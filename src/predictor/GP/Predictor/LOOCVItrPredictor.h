/*
 * LOOCVItrPredictor.h
 *
 *  Created on: Aug 8, 2014
 *      Author: zhoujingbo
 */

#ifndef LOOCVITRPREDICTOR_H_
#define LOOCVITRPREDICTOR_H_

#include "LOOCVPredictor.h"


/**
 * TODO:
 * implement the iterative multistep ahead prediction.
 * implement two interface of Predictor
 *          (1) virtual void inferItrPrediction()=0;
 *          (2) virtual void updateNextXtrn() = 0;
 *
 *     inherit graph: Predictor=>MetaGPPredictor=>LOOCVPredictor (i.e. LOOCVDirPredictor) =>LOOCVItrPredictor
 *     					  \\
 *     						== RegPredictor
 *     Refer to:Girard, A., C. E. Rasmussen, J. Quinonero-Candela, and R. Murray-Smith. “Gaussian Process Priors with Uncertain Inputs -  Application to Multiple-Step Ahead Time Series Forecasting.” 2003
 *         	    Girard, Agathe, and Roderick Murray-smith. “Gaussian Process: Prediction at a Noisy Input and Application to Iterative Multiple-Step Ahead Forecasting of Time-Series.” In In: Switching and Learning in Feedback Systems, Eds. Springer, n.d.
 *
 *
 */

template <class T>
class LOOCVItrPredictor: public LOOCVPredictor<T> {
public:
	//LOOCVItrPredictor(){}

	LOOCVItrPredictor(vector<int>& Lvec, vector<int>& Kvec,
				const vector<vector<vector<T> > >& dataXtrn,
				const vector<vector<T> >& dataYtrn, const vector<T> & dataXtst,
				double range, double sill, double nugget) :
					LOOCVPredictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst, range,
						sill, nugget){

	}

	LOOCVItrPredictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<T> & dataXtst,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<vector<T> >& weight,
			double range, double sill, double nugget) :
				LOOCVPredictor<T>(Lvec, Kvec, dataXtrn, dataYtrn, dataXtst, weight,
						range, sill, nugget) {

	}

	virtual ~LOOCVItrPredictor(){}



	/**
		 * make perdiction based on the Xtrn and Ytrn, the Xtrn and Ytrn should have been configured.
		 * This also use error propagation for iterative prediction
		 */
		void inferItrPrediction() {

			for (int i = 0; i < this->numL; i++) {

				mat& XtrnSlice = this->Xtrn[i];
				vec& YtrnSlice = this->Ytrn[i];
				vec& weightSlice = this->weightVec[i];

				int dim = this->Lvec[i];

				for (int j = 0; j < this->numK; j++) {
					mat XtrnSgm = XtrnSlice.rows(0, this->Kvec[j] - 1);
					vec YtrnSgm = YtrnSlice.rows(0, this->Kvec[j] - 1);
					vec weightSgm = weightSlice.rows(0, this->Kvec[j] - 1);

					vec meansgm(1); // = vec().zeros(1);
					vec varSgm(1); // = vec().zeros(1);

					if (NULL == this->PrdMat[i][j] || this->iterativeSteps == 0) {
						this->PrdMat[i][j] = new LOOCVGP(dim, 1, XtrnSgm, YtrnSgm,
								this->sumCFMat[i][j], weightSgm);
					}

					inferItrPredictionSgm(this->PrdMat[i][j], XtrnSgm, YtrnSgm,
							this->XtstMeanItr[i][j],this-> XtstVarItr[i][j],
							*(this->gaussianCFMat[i][j]), meansgm, varSgm);

					assert(meansgm.n_rows == 1 && varSgm.n_rows == 1);

					this->mean[i][j].insert_rows(this->iterativeSteps, meansgm);
					this->var[i][j].insert_rows(this->iterativeSteps, varSgm);

				}
			}

			this->iterativeSteps++;
		}

		void inferItrPredictionSgm(MetaGP* metaGP, mat& XtrnSgm, vec& YtrnSgm,
				vec& _XtstMeanItrSgm, mat& _XtstVarItrSgm,
				CovarianceFunction& _gaussianCF, vec & _meanSgm, vec& _varSgm){

			metaGP->setXYtrn(XtrnSgm, YtrnSgm);
			SCGModelTrainer scggp(*metaGP);
			scggp.setDisplay(this->display);
			scggp.Train(0);//UtlPredictor_namespace::trainStep);//for exp

			metaGP->makeIterativePredictions(_XtstMeanItrSgm, _XtstVarItrSgm,
					_gaussianCF, _meanSgm, _varSgm);

		}


		//for direct prediction, nothing to do with Xtrn
		//for iterative prediction, call "shiftConcatXtrn()"
		//implement for different prediction functions
		virtual void updateNextXtrn() {

			this->shiftConcatXtrn();

		}



};

#endif /* LOOCVITRPREDICTOR_H_ */
