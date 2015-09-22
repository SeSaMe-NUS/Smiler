/*
#include "psgp_settings.h"
#include "psgp_data.h"
#include "psgp_estimator.h"

#define length Rf_length
#define allocVector Rf_allocVector

*
 * Interface between R and C++.
 * This provides 2 wrapper functions which format the R data pointers to PSGP-compatible
 * objects and pointers.


#define NUM_VARIOGRAM_PARAMETERS 5

extern "C" {

*
 * Populate a PsgpData object with the data from R structures

PsgpData prepareData(SEXP xData, SEXP yData, SEXP params, SEXP sensorMetadata, SEXP sensorIndices, bool paramsFromVario) {
	PsgpData data;
	data.setX(xData);
	data.setY(yData);

	// If variogram parameters are passed in, use them to set PSGP parameters
	// If none are passed in, they won't be used (they are only used for parameter
	// estimation, not for prediction)
	if (paramsFromVario) {
		data.setPsgpParamsFromVariogram(params);
	} else {
		data.setPsgpParamsFromInference(params);
	}
	data.setSensorMetadata(sensorIndices, sensorMetadata);
	return data;
}

*
 * Estimate PSGP parameters
 * We use the variogram parameters as a first guess

SEXP estimateParams(SEXP xData, SEXP yData, SEXP vario, SEXP sensorIndices,
		SEXP unusedIndices, SEXP sensorMetadata) {

	double *varioPtr = REAL(vario);   // Pointer to variogram parameters

	// PSGP parameters in R format
	SEXP R_psgpParams;
	PROTECT( R_psgpParams = allocVector(REALSXP, NUM_PSGP_PARAMETERS) );

	// Create and allocate pointer to PSGP parameter vector
	double* psgpParams = REAL(R_psgpParams);
	UNPROTECT(1);

	// Copy current variogram parameters to parameter array
	memcpy(psgpParams, varioPtr, NUM_VARIOGRAM_PARAMETERS * sizeof(double));

	// Convert data from R structures to vectors and matrices
	// TODO: CHECK: errorIndices was used in the original code instead of sensorIndices
	PsgpData data = prepareData(xData, yData, vario, sensorMetadata, sensorIndices, true);

	// Estimate parameters.
	// This also updates the parameter values in psgpParameters
	// and in variogramParameters (if the initial parameters were not
	// valid, i.e. negative...)
	PsgpEstimator estimator;
	vec params;
	estimator.learnParameters(data, params);

	// Copy final parameters over to psgpParameters
	for(int i=0; i<params.n_elem; i++)
	{
	    *psgpParams++ = params(i);
	}

	// Add padding zeros (remember psgpParameters has fixed size and is
	// likely to be bigger than we need)
	for(int i=params.n_elem; i<NUM_PSGP_PARAMETERS; i++)
	{
	    *psgpParams++ = 0.0;
	}

	return R_psgpParams;
}


*
 * Predict at a new set of inputs xPred. Psgp parameters are passed in
 * from R as R_psgpParams.

SEXP predict(SEXP xData, SEXP yData, SEXP xPred, SEXP R_psgpParams, SEXP sensorIndices,
		SEXP unusedIndices,  SEXP sensorMetadata) {

	SEXP meanResult;
	SEXP varResult;
	SEXP ans;

	// Convert data from R structures to vectors and matrices
	PsgpData data = prepareData(xData, yData, R_psgpParams, sensorMetadata, sensorIndices, false);

	vec psgpParams(REAL(R_psgpParams), length(R_psgpParams));

	// Prediction inputs and outputs
	int numPred = length(xPred)/2;
	mat Xpred(REAL(xPred), numPred, 2);

	vec meanPred(numPred);
	vec varPred(numPred);

	// Make predictions using PSGP
	PsgpEstimator estimator;
	Rprintf("Make prediction\n");
	estimator.makePredictions(data, psgpParams, Xpred, meanPred, varPred);

	// Copy results to R structures
	PROTECT(meanResult = allocVector(REALSXP, numPred));
	PROTECT(varResult = allocVector(REALSXP, numPred));
	PROTECT(ans = allocVector(VECSXP, 2));

	double* ptr_meanResult = REAL(meanResult);
	double* ptr_varResult = REAL(varResult);
	for(int i=0; i < numPred; i++)
	{
		ptr_meanResult[i] = meanPred(i);
		ptr_varResult[i] = varPred(i);
	}

	SET_VECTOR_ELT(ans, 0, meanResult);
	SET_VECTOR_ELT(ans, 1, varResult);

	UNPROTECT(3);
	return ans;
}
} // END OF EXTERN C


*/
