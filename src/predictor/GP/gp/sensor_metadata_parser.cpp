/*

 * sensor_metadata_parser.cpp
 *
 *  Created on: 22 Jan 2012
 *      Author: barillrl


#include "sensor_metadata_parser.h"
*
 * Constructor
 * The default variance parameter is used for unsupported sensor models
 * if we cannot estimate it from the supported sensor models.

SensorMetadataParser::SensorMetadataParser(double defaultVariance) {
	resetModelStats();
	this->defaultVariance = defaultVariance;
}

SensorMetadataParser::~SensorMetadataParser() {
	// TODO Auto-generated destructor stub
}

*
 * Convert the R metadata to a list of metadata strings and return the
 * corresponding vector of likelihood models.
 * A same model (metadata entry) might be used for several observations. Its
 * index will appear several times in R_indices.

vector<LikelihoodType*> SensorMetadataParser::parseMetadata(SEXP R_indices, SEXP R_metadata, arma::ivec &indices) {

	unsigned int numModels = length(R_metadata);
	vector<string> metadata;

	// Shift indexes (starting from 1 in data, from 0 in Armadillo)
	int* indexPtr = INTEGER(R_indices);
	indices = arma::ivec(indexPtr, numModels);
	indices -= 1;

	// The metadata table, if provided, is terminated by an empty line
	// Need to remove it from the size.
	if (numModels > 0) numModels--;

	// we need to make a table of pointers to the sensor model strings
	metadata = std::vector<string>(numModels);

	// Convert R metadata to list of string
	for (unsigned int i = 0; i < numModels; i++) {
		metadata[i] =
				string(
						const_cast<char*>(CHAR(STRING_ELT(VECTOR_ELT(R_metadata, (int)i),0))));
	}

	return parseMetadata(metadata);
}

*
 * Converts a list of sensor metadata (string) to a list of likelihood
 * models.

vector<LikelihoodType*> SensorMetadataParser::parseMetadata(
		vector<string> metadata) {
	vector<LikelihoodType*> sensorModels(metadata.size());

	unsigned int numModels = sensorModels.size();

	resetModelStats();

	// Create a likelihood model for each metadata statement
	for (unsigned int i = 0; i < numModels; i++) {
		string metadatum = metadata[i];
		sensorModels[i] = getLikelihoodFor(metadatum);
	}

	// Validate the likelihood models - if invalid models (i.e. with wrong
	// sensor type) are passed in, they will be replaced with a default
	// Gaussian model with variance inferred from the other valid models.
	validateModels(sensorModels);

	return sensorModels;
}

*
 * Return an instance of LikelihoodType* for a given metadatum
 * Datums should be of the form "modelname, param1, param2, ..."

LikelihoodType* SensorMetadataParser::getLikelihoodFor(string metadatum) {
	// Rprintf("Noise model: %s\n", metadatum.c_str());

	// Split string in 2 w.r.t. to first (non-trailing) comma delimiter
	std::vector<string> tokens;
	tokenise(metadatum, tokens, ", ", 1);

	// Retrieve distribution name
	string modelName = tokens[0];
	vec modelParams;

	// Use vector initialisation from string - if this does not work,
	// i.e. string cannot be converted to double, flag the noise model as
	// invalid by setting the modelIndex to -1
	try {
		modelParams = vec(formatParams(tokens[1]));
	} catch (std::runtime_error &e) {
		Rprintf("** Error in metadata parsing for noise model %s:",
				modelName.c_str());
		Rprintf("   Invalid parameter string \"%s\"", tokens[1].c_str());
		Rprintf("   Parameter string must be a sequence of numeric values");
		Rprintf("   separated by commas, e.g. \"1.23,4,5.6,78.9\"");

		// Flag observation as not having a valid model
		modelName = INVALID_MODEL_NAME;
	}

	return getLikelihoodByName(modelName, modelParams);
}

*
 * Returns an instance of LikelihoodType* corresponding to the
 * model name specified and with parameters set from modelParams

LikelihoodType* SensorMetadataParser::getLikelihoodByName(string modelName,
		vec modelParams) {
	LikelihoodType* likelihoodModel = NULL;

	// If a valid model name/params exist, create likelihood model
	if (modelName == "GAUSSIAN") {
		
		// TODO: At the moment, the GaussianLikelihood takes only the variance parameter
		// Change to GaussianLikelihood(mean, variance)
		// Also, add support for LikelihoodType(Col params) to make it more generic
		likelihoodModel = new GaussianLikelihood(modelParams(1));

		// Update overall stats - these are used to infer the parameters of
		// a default Gaussian likelihood to replace invalid models if found.
		averageModelVariance += modelParams(1);
		gaussianModelCount++;
	} else {
		Rprintf("Unrecognized observation noise model: %s\n",
				modelName.c_str());
		invalidModelCount++;
	}

	return likelihoodModel;
}

*
 * Validate the likelihood models.
 * If some unsupported sensor models were found, they have been replaced
 * by an invalid model (NULL). We now find these and replace them with
 * a Gaussian model with parameters estimated from the valid Gaussian models
 * found, or with some default parameters if no valid models were found.

void SensorMetadataParser::validateModels(
		std::vector<LikelihoodType*> &models) {

	// Return if no invalid model found
	if (invalidModelCount == 0)
		return;

	// If we found some valid Gaussian noise models, infer parameters for
	// default likelihood from theirs (take average mean and average variance).
	// Should hopefuly be slightly better than our first guess.
	if (gaussianModelCount > 1) {
		averageModelMean /= gaussianModelCount;
		averageModelVariance /= gaussianModelCount;
	}

	// If no valid Gaussian model is found, revert to the default Gaussian
	// likelihood
	else {
		averageModelVariance = defaultVariance;
	}

	Rprintf(
			"%d observations without a valid/supported noise model were found. ",
			invalidModelCount);
	Rprintf("These will be given a default Gaussian noise model with ");
	Rprintf("(mean, variance) = (%f, %f)\n", averageModelMean,
			averageModelVariance);

	// Attribute all observations with an invalid noise model the default
	// (Gaussian) likelihood model
	for (vector<LikelihoodType*>::iterator model = models.begin();
			model < models.end(); model++) {
		if (*model == NULL) {
			(*model) = new GaussianLikelihood(averageModelVariance);
		}
	}
}

*
 * String tokeniser - split a string into a vector of substring identified
 * by a delimiter (space by default). Optionally, stop after a certain number
 * of tokens has been found and return the rest of the string as the last token
 *
 * Adapted from: http://oopweb.com/CPP/Documents/CPPHOWTO/Volume/C++Programming-HOWTO-7.html

void SensorMetadataParser::tokenise(const std::string& str,
		std::vector<std::string>& tokens, const std::string& delimiters,
		unsigned int stopAfter) {
	// by default, do not stop after a given number of tokens (i.e. process
	// the whole string)
	if (stopAfter == 0)
		stopAfter = str.size() + 1;

	// Find start and end of first token
	std::string::size_type tokenStart = str.find_first_not_of(delimiters, 0); // First char
	std::string::size_type tokenEnd = str.find_first_of(delimiters, tokenStart); // Following delimiter

	while ((std::string::npos != tokenEnd || std::string::npos != tokenStart)
			&& tokens.size() < stopAfter) {
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(tokenStart, tokenEnd - tokenStart));

		// Move to next token
		tokenStart = str.find_first_not_of(delimiters, tokenEnd);
		tokenEnd = str.find_first_of(delimiters, tokenStart);
	}

	// If we have stopped after a given number of tokens, add the rest of the
	// string as a final token
	if (tokens.size() == stopAfter) {
		tokens.push_back(str.substr(tokenStart));
	}
}

*
 * Reset the model stats

void SensorMetadataParser::resetModelStats() {
	gaussianModelCount = 0;
	invalidModelCount = 0;
	averageModelMean = 0.0;
	averageModelVariance = 0.0;
}

*
 * Replaces comma with spaces in a string
 * Note: Armadillo allows space-separated strings of numbers to initialise
 * a vector, but the metadata has a comma-separated list of values.

string SensorMetadataParser::formatParams(string params) {
	string delimFrom = ",";
	string delimTo = " ";

	// string::length clashes with R's length (even when using namespace)
	// so we specify the length of the delimiter manually.
	int delimLen = 1;

	size_t match = params.find(delimFrom);
	while (match != string::npos) {
		params.replace(match, delimLen, delimTo);
		match = params.find(delimFrom);
	};
	return params;
}
*/
