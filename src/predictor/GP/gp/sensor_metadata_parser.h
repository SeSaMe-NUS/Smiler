/*

 * sensor_metadata_parser.h
 *
 *  Created on: 22 Jan 2012
 *      Author: barillrl


#ifndef SENSOR_METADATA_PARSER_H_
#define SENSOR_METADATA_PARSER_H_

#include <vector>
#include <string>

#include "likelihoodModels/LikelihoodType.h"
#include "likelihoodModels/GaussianLikelihood.h"
#include "psgp_common.h"

#include "Rinternals.h"
#define length Rf_length
#define allocVector Rf_allocVector

// ID for invalid noise model (when using observation noise)
#define INVALID_MODEL_NAME "INVALID_MODEL"

class SensorMetadataParser {
public:
	SensorMetadataParser(double defaultVariance);
	virtual ~SensorMetadataParser();

	vector<LikelihoodType*> parseMetadata(SEXP R_indices, SEXP R_metadata, arma::ivec &indices);
	vector<LikelihoodType*> parseMetadata(vector<string> metadata);

private:
	int gaussianModelCount;        // Keeps track of Gaussian likelihood models found
	double averageModelVariance;   // and their average variance
	double averageModelMean;       // The mean is ignored at the moment (no bias in likelihood type)
	double defaultVariance;        // Default model variance for unsupported likelihood models (used only as a last resort)
	int invalidModelCount;         // Keeps track of invalid/unsupported models

	void tokenise(const std::string& str, std::vector<std::string>& tokens, const std::string& delimiters = " ", unsigned int stopAfter=0);
	LikelihoodType* getLikelihoodFor(string metadatum);
	LikelihoodType* getLikelihoodByName(string modelName, vec modelParams);
	void validateModels( std::vector<LikelihoodType*> &models );
	void resetModelStats();
	string formatParams(string params);
};

#endif  SENSOR_METADATA_PARSER_H_
*/
