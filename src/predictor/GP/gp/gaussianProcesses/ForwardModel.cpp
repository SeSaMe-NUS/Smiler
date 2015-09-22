#include "ForwardModel.h"

ForwardModel::ForwardModel(int Inputs, int Outputs)
{
	inputDimensions = Inputs;
	outputDimensions = Outputs;
}

ForwardModel::~ForwardModel()
{
}

int ForwardModel::getInputDimensions() const
{
	return inputDimensions;
}

int ForwardModel::getOutputDimensions() const
{
	return outputDimensions;
}
