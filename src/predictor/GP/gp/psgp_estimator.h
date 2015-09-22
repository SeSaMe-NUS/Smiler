/*

 * psgp_estimator.h
 *
 *  Created on: 21 Jan 2012
 *      Author: barillrl


#ifndef PSGP_ESTIMATOR_H_
#define PSGP_ESTIMATOR_H_

#include "psgp_data.h"
#include "psgp_settings.h"
#include "gaussianProcesses/PSGP.h"
#include "likelihoodModels/GaussianLikelihood.h"
#include "covarianceFunctions/CovarianceFunction.h"
#include "covarianceFunctions/SumCovarianceFunction.h"
#include "covarianceFunctions/ExponentialCF.h"
#include "covarianceFunctions/Matern5CF.h"
#include "covarianceFunctions/ConstantCF.h"
#include "covarianceFunctions/WhiteNoiseCF.h"
#include "optimisation/SCGModelTrainer.h"

class PsgpEstimator {
public:
	PsgpEstimator();
	virtual ~PsgpEstimator();

	void learnParameters(PsgpData &data, vec &psgpParams);
	void makePredictions(PsgpData &data, vec psgpParams, mat Xpred, vec &meanPred, vec &varPred);

	void setFixedSweeps(int n) { fixedSweeps = n; }
	void setUpdateSweeps(int n) { updateSweeps = n; }
	void setActivePoints(int n) { activePoints = n; }

private:
	unsigned int fixedSweeps;
	unsigned int updateSweeps;
	unsigned int activePoints;

	ExponentialCF* expKernel;
	Matern5CF* mat5Kernel;
	ConstantCF* constKernel;
	WhiteNoiseCF* nuggetKernel;

	CovarianceFunction* covFun;
	PSGP* psgp;

	void setupCovarianceFunction(const PsgpData &data, bool forPrediction);
	void setupPsgp(PsgpData &data, bool forPrediction);
};

#endif  PSGP_ESTIMATOR_H_
*/
