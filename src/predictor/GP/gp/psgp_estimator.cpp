/*

 * psgp_estimator.cpp
 *
 *  Created on: 21 Jan 2012
 *      Author: barillrl


#include "psgp_estimator.h"

PsgpEstimator::PsgpEstimator() {
	fixedSweeps = NUM_SWEEPS_FIXED;
	updateSweeps = NUM_SWEEPS_CHANGING;
	activePoints = MAX_ACTIVE_POINTS;
	covFun = NULL;
	psgp = NULL;
}

PsgpEstimator::~PsgpEstimator() {
	if (covFun != NULL) {
		delete expKernel;
		delete constKernel;
		delete mat5Kernel;
		delete nuggetKernel;
		delete covFun;
	}

	if (psgp != NULL) {
		delete psgp;
	}

}

void PsgpEstimator::learnParameters(PsgpData& data, vec &psgpParams) {

	setupPsgp(data, false);

	covFun->displayCovarianceParameters();

	SCGModelTrainer gpTrainer(*psgp); // SCG optimisation method
	gpTrainer.setAnalyticGradients(true);
	gpTrainer.setCheckGradient(false);

	Rprintf("Finding optimal parameters");
	for (int i = 0; i < PSGP_PARAM_ITERATIONS; i++) {
		gpTrainer.Train(PSGP_SCG_ITERATIONS);
		psgp->recomputePosterior();
	}

	// Copy final parameters over to params
	psgpParams = covFun->getParameters();
}

*
 * Compute the predictive mean and variance (without correlation) at the
 * new inputs Xpred. The meanPred and varPred vectors should have been
 * allocated prior to calling this function.

void PsgpEstimator::makePredictions(PsgpData &data, vec psgpParams, mat Xpred, vec &meanPred,
		vec &varPred) {

	// Create PSGP without a nugget kernel
	// This is the default setup - covariance parameters are
	// overriden below with the provided values
	setupPsgp(data, true);

	// Override covariance function parameters
	covFun->setParameters(psgpParams);

	if (!USING_CHUNK_PREDICTION) {
		Rprintf("Predicting...");
		psgp->makePredictions(meanPred, varPred, Xpred, *covFun);
	} else {
		int numPred = Xpred.n_rows;
		int startVal = 0;
		int chunkSize = CHUNK_SIZE;
		int endVal = chunkSize - 1;

		// Last chunk might be smaller than CHUNK_SIZE
		if (endVal > numPred) {
			endVal = numPred - 1;
			chunkSize = endVal - startVal + 1;
		}

		// Predict one chunk at a time
		while (startVal < numPred) {
			Rprintf("  Predicting chunk [ %d:%d / %d ]\n", startVal, endVal,
					numPred);

			mat XpredChunk = Xpred.rows(startVal, endVal);

			// Predicted mean and variance for data chunk
			vec meanPredChunk(chunkSize);
			vec varPredChunk(chunkSize);

			Rprintf("Predict using PSGP\n");
			psgp->makePredictions(meanPredChunk, varPredChunk, XpredChunk,
					*covFun);

			meanPred.rows(startVal, startVal + chunkSize - 1) = meanPredChunk;
			varPred.rows(startVal, startVal + chunkSize - 1) = varPredChunk;

			startVal = endVal + 1;
			endVal = endVal + chunkSize;

			if (endVal >= numPred) {
				endVal = numPred - 1;
				chunkSize = endVal - startVal + 1;
			}
		} // end while loop
	}

	Rprintf("PSGP used the following parameters:");
	covFun->displayCovarianceParameters();
}


*
 * Instantiate and initialise the PSGP object

void PsgpEstimator::setupPsgp(PsgpData &data, bool forPrediction) {

	setupCovarianceFunction(data, forPrediction);

	// Create PSGP instance
	psgp = new PSGP(data.getX(), data.getY(), *covFun, activePoints,
			updateSweeps, fixedSweeps);

	// Set up the likelihood - Gaussian by default, unless
	// sensor models have been provided
	GaussianLikelihood* defaultLikelihood;

	int numModels = data.getSensorModels().size();

	if (numModels == 0) {
		Rprintf("No noise model specified\n");
		Rprintf("Defaulting to GAUSSIAN with variance %f\n",
				(data.getNugget() * LIKELIHOOD_NUGGET_RATIO));
		defaultLikelihood = new GaussianLikelihood(
				data.getNugget() * LIKELIHOOD_NUGGET_RATIO);

		// Compute initial posterior with default likelihood
		psgp->computePosterior(*defaultLikelihood);
	} else {
		Rprintf("Observation error characteristics specified.\n");
		Rprintf("Building error models from sensor metadata table.\n");

		arma::ivec sensorIndices = data.getSensorIndices();
		psgp->computePosterior(sensorIndices, data.getSensorModels());
	}

	// Use approximate objective function (more stable)
	psgp->setLikelihoodType(Approximate);
}

*
 * Set up the covariance function

void PsgpEstimator::setupCovarianceFunction(const PsgpData &data, bool forPrediction) {
	// Covariance function components
	expKernel = new ExponentialCF(data.getRangeExp(), data.getSillExp());
	mat5Kernel = new Matern5CF(data.getRangeMat5(), data.getSillMat5());
	constKernel = new ConstantCF(data.getBias());
	nuggetKernel = new WhiteNoiseCF(data.getNugget());

	// Final covariance function is kernel + bias + white noise
	covFun = new SumCovarianceFunction(*expKernel);
	((SumCovarianceFunction*) covFun)->addCovarianceFunction(*mat5Kernel);
	((SumCovarianceFunction*) covFun)->addCovarianceFunction(*constKernel);

	if (!forPrediction) {
		((SumCovarianceFunction*) covFun)->addCovarianceFunction(*nuggetKernel);
	}
}

*/
