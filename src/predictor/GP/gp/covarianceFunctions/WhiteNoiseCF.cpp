#include "WhiteNoiseCF.h"

using namespace psgp_arma;
WhiteNoiseCF::WhiteNoiseCF(double var) : CovarianceFunction("White noise")
{
	numberParameters = 1;
	setDefaultTransforms();
	variance = var;


}

WhiteNoiseCF::~WhiteNoiseCF()
{

}

/* RB: This is not quite right - the diagonal elements should return variance, 
 * not zero! This function really ought to know the row/column for the element...
 * A quick (but unsafe!) fix is to check if A == B (this should be a diagonal 
 * element, unless we have twice the same point in the data set, which should not
 * happen - in theory!)   
 */
inline double WhiteNoiseCF::computeElement(const vec& A, const vec& B) const
{
//	if (A==B) 
//	    return variance;
//	else
		return 0.0;
}

inline double WhiteNoiseCF::computeDiagonalElement(const vec& A) const
{
	return variance*variance;

}

double WhiteNoiseCF::getParameter(unsigned int parameterNumber) const
{
	

	switch(parameterNumber)
	{
		case 0 : return(variance);
					break;
		default: break;
	}
	Rprintf("Warning: should not have reached here in WhiteNoiseCF::getParameter");
	return(0.0);
}

void WhiteNoiseCF::setParameter(unsigned int parameterNumber, const double value)
{
	

	switch(parameterNumber)
	{
		case 0 : variance = value;

					break;
		default: break;
	}
}

string WhiteNoiseCF::getParameterName(unsigned int parameterNumber) const
{
	

	switch(parameterNumber)
	{
		case 0 : return("Variance");
					break;
		default: break;

	}
	return("Unknown parameter");
}

void WhiteNoiseCF::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{
	

	Transform* t = getTransform(parameterNumber);
	double gradientModifier = t->gradientTransform(getParameter(parameterNumber));

	switch(parameterNumber)
	{
		case 0 :
		{
			computeSymmetric(PD, X);
			PD *=  (2*gradientModifier / variance);
			return;
			break;
		}
	}
    Rprintf("Warning: should not have reached here in GaussianCF::getParameterPartialDerivative");
}
