/*

 * psgp_data.h
 *
 *  Created on: 21 Jan 2012
 *      Author: barillrl


#ifndef PSGP_DATA_H_
#define PSGP_DATA_H_

#include <algorithm>
#include <cstdlib>

#include "psgp_common.h"
#include "psgp_settings.h"
#include "likelihoodModels/LikelihoodType.h"
#include "sensor_metadata_parser.h"

#include "Rinternals.h"

#define length Rf_length
#define allocVector Rf_allocVector

class PsgpData {
public:
	PsgpData();
	virtual ~PsgpData();

	void setX(const SEXP xPtr);
	void setY(const SEXP yPtr);
	void setPsgpParams(double range, double sill, double nugget, double bias);
	void setPsgpParamsFromVariogram(const SEXP varioParams);
	void setPsgpParamsFromInference(const SEXP psgpParams);
	void setSensorMetadata(SEXP indices, SEXP metadata);

	void setX(const mat X);
	void setY(const vec Y);
	void setSensorModels(arma::ivec indices, const std::vector<LikelihoodType*> sensorModels);

	mat& getX() { return X; }
	vec& getY() { return Y; }

	int getObsCount();
	double getBias() const;
    double getNugget() const;
    double getRangeExp() const;
    double getRangeMat5() const;
    double getSillExp() const;
    double getSillMat5() const;

    std::vector<LikelihoodType*> getSensorModels() const;

    arma::ivec getSensorIndices();

	// Set the maximum number of observations
	// void setObsLimit(int n) { obsLimit = n; }

private:
	// int obsLimit;

	double rangeExp;    // Parameters of exponential kernel
	double sillExp;
	double rangeMat5; 	// Parameters of matern kernel
	double sillMat5;
	double nugget;		// Variance of nugget (white noise) kernel
	double bias;		// Variance of constant (bias) kernel

	// List of sensor models
	std::vector<LikelihoodType*> sensorModels;
	// Indices indicating which model is used for which observations
	arma::ivec sensorIndices;


	// Matrix to store the data (2 column input + 1 column output = 3 columns)
	// mat data;
	mat X;
	vec Y;

	// If not using all the observations, a random sample is selected
	// according to a random set of indices
	// bool shuffled;
	// uvec shuffledIndices;

	// void shuffleObs();
	LikelihoodType* getLikelihoodType(string modelMetadata);
};

#endif  PSGP_DATA_H_
*/
