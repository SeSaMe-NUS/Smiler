/*


 * psgp_data.cpp
 *
 *  Created on: 21 Jan 2012
 *      Author: barillrl


#include "psgp_data.h"

PsgpData::PsgpData() {
	// Some default values for parameters
	// These are meant to be overridden but are provided so that
	// the covariance function in PsgpEstimator can be initialised
	// from the PsgpData prior to settings its parameters
	rangeExp = 1.0;
	sillExp = 1.0;
	rangeMat5 = 1.0;
	sillMat5 = 1.0;
	nugget = 0.01;
	bias = 0.0;
}

PsgpData::~PsgpData() {
}

*
 * Set the inputs.
 * Sets the first 2 columns of the data matrix using the
 * 2D input data from the provided pointer.

void PsgpData::setX(const SEXP xPtr) {
	// Reshape data as a 2-column matrix (coordinate points)
	double* xData = REAL(xPtr);
	int nobs = length(xPtr) / 2;
	mat X(xData, nobs, 2);
	setX(X);
}

void PsgpData::setX(const mat X) {
	this->X = X;
//	if (data.n_rows != X.n_rows) {
//		data.resize(X.n_rows, 3);
//	}
//	data.cols(0, 1) = X;
}


*
 * Set the output.
 * Sets the 3rd column of the data matrix using the output data
 * from the provided pointer.

void PsgpData::setY(const SEXP yPtr) {
	// Reshape data as a 2-column matrix (coordinate points)
	double* yData = REAL(yPtr);
	int nobs = length(yPtr);
	vec Y(yData, nobs);
	setY(Y);
}


void PsgpData::setY(const vec Y) {
	this->Y = Y;
//	if (data.n_rows != Y.n_rows) {
//		data.resize(Y.n_rows, 3);
//	}
//	data.col(2) = Y;
}


*
 * Set the PSGP parameters from R-estimated variogram parameters

void PsgpData::setPsgpParamsFromVariogram(const SEXP varioParams) {
	// RB: get variogram model parameters as a starting point
	// varioParams[0] is the model ID - we can ignore it.
	double* vario = REAL(varioParams);
	double range = vario[1];
	double sill = vario[2];
	nugget = vario[3];

	// Empirical estimation of bias (if any obs available)
	bias = 0.01;

	if (Y.n_rows > 0) {
		double ymean = mean(Y);
		if (ymean != 0.0) {
			bias = fabs(1.0/ymean);
		}
	}

	// Initialise both Exp and Matern5 kernels to the same sill and range
	setPsgpParams(range, sill, nugget, bias);
}

*
 * Set the PSGP parameters from PSGP-inferred parameters
 * Remember these are log transformed

void PsgpData::setPsgpParamsFromInference(const SEXP psgpParams) {
	vec params =  vec(REAL(psgpParams), length(psgpParams));
	rangeExp = exp(params(0));
	sillExp = exp(params(1));
	rangeMat5 = exp(params(2));
	sillMat5 = exp(params(3));
	bias = exp(params(4));
	nugget = exp(params(5));
}

*
 * Set PSGP (covariance) parameters

void PsgpData::setPsgpParams(double _range, double _sill, double _nugget,
		double _bias) {

	double range = _range;
	double sill = _sill;
	nugget = _nugget;
	bias = _bias;

	// Make sure everything is fine - the variogram estimation can
	// give invalid parameters (negative range...) - if so, revert
	// to some first guess from the data.
	if (range <= 0.0 || sill <= 0.0 || nugget <= 0.0) {
		Rprintf("Invalid parameters: either the range, sill or nugget\n");
		Rprintf("is negative or zero. Reverting to defaults.\n");

		double r1 = abs(arma::min(vec(arma::max(X, 0) - arma::min(X, 0))));
		double r2 = abs(arma::min(vec(arma::max(X, 1) - arma::min(X, 1))));

		range = 0.25 * ((r1 + r2) / 2.0);
		sill = abs(arma::var(Y));
		nugget = 0.5 * sill;
	}

	rangeExp = range;
	sillExp = sill;
	rangeMat5 = range;
	sillMat5 = sill;

	Rprintf("Range: %f\n", range);
	Rprintf("Sill: %f\n", sill);
	Rprintf("Nugget: %f\n", nugget);
	Rprintf("Bias: %f\n", bias);
}

//
///**
// * Return a random subset of X with at most the limit of rows
//
//mat PsgpData::getTruncatedData() {
//	if (!shuffled) {
//		shuffleObs();   // Randomize the data
//	}
//
//	// Return at most obsLimit rows of data
//	int n = std::min(obsLimit, (int) data.n_rows);
//	return data.rows(0, n - 1);
//}
//

double PsgpData::getBias() const {
	return bias;
}

double PsgpData::getNugget() const {
	return nugget;
}

int PsgpData::getObsCount()
{
	
	return X.n_rows;
}

void PsgpData::setSensorModels(arma::ivec sensorIndices, const std::vector<LikelihoodType*> sensorModels)
{
	this->sensorModels = sensorModels;
	this->sensorIndices = sensorIndices;
}

std::vector<LikelihoodType*> PsgpData::getSensorModels() const {
	return sensorModels;
}

*
 * Return the array of shuffled observation indices
 * At the moment, no shuffling so this is a direct mapping

arma::ivec PsgpData::getSensorIndices() {
	return sensorIndices;
}

*
 * Shuffle the observations


void PsgpData::shuffleObs() {
	int nobs = X.n_rows;
	shuffledIndices = psgp_arma::randperm(nobs);
	mat shuffledData(nobs, data.n_cols);

	// Copy rows to new matrix in shuffled order
	for (int i=0; i<data.n_rows; i++) {
		shuffledData.row(i) = data.row( shuffledIndices[i] );
	}
	data = shuffledData;
	shuffled = true;
}


*
 * Create sensor model table from metadata pointer and indices specified
 * (sensor metadata might not be in the same order as observations, indices
 * provides the mapping).

void PsgpData::setSensorMetadata(SEXP indices, SEXP metadata) {
	SensorMetadataParser parser(nugget * LIKELIHOOD_NUGGET_RATIO);
	sensorModels = parser.parseMetadata(indices, metadata, sensorIndices);
}

double PsgpData::getRangeExp() const
{
    return rangeExp;
}

double PsgpData::getRangeMat5() const
{
    return rangeMat5;
}

double PsgpData::getSillExp() const
{
    return sillExp;
}

double PsgpData::getSillMat5() const
{
    return sillMat5;
}



*/

