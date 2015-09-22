#include "ConstantCF.h"

ConstantCF::ConstantCF(double amp) : CovarianceFunction("Constant")
{
	numberParameters = 1;
	amplitude = amp;
	setDefaultTransforms();
}

ConstantCF::~ConstantCF()
{

}

inline double ConstantCF::computeElement(const vec& A, const vec& B) const
{
	return 1.0 / amplitude;
}

inline double ConstantCF::computeDiagonalElement(const vec& A) const
{
	return 1.0 / amplitude;
}

double ConstantCF::getParameter(unsigned int parameterNumber) const
{
	

	switch(parameterNumber)
	{
		case 0 : return(amplitude);
					break;
		default: break;
	}
	Rprintf("Warning: should not have reached here in ConstantCF::getParameter");
	return(0.0);
}

void ConstantCF::setParameter(unsigned int parameterNumber, const double value)
{
	

	switch(parameterNumber)
	{
		case 0 : amplitude = value;
					break;
		default: break;
	}
}

string ConstantCF::getParameterName(unsigned int parameterNumber) const
{
	

	switch(parameterNumber)
	{
		case 0 : return("Amplitude");
					break;
		default: break;

	}
	return("Unknown parameter");
}

void ConstantCF::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{
	

	Transform* t = getTransform(parameterNumber);
	double gradientModifier = t->gradientTransform(getParameter(parameterNumber));

	switch(parameterNumber)
	{
		case 0 :
		{
		    PD = -gradientModifier/(amplitude*amplitude) * arma::ones(X.n_rows, X.n_rows);
			return;
		}
	}
	Rprintf("Warning: should not have reached here in ConstantCF::getParameterPartialDerivative");
}
