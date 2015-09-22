#include "ExponentialCF.h"

using namespace psgp_arma;

ExponentialCF::ExponentialCF(double lengthscale, double var) : CovarianceFunction("Isotropic Exponential")
{
	numberParameters = 2;
	setDefaultTransforms();

	range = lengthscale;
	variance = var;
}

ExponentialCF::ExponentialCF(vec parameters) : CovarianceFunction("Isotropic Exponential")
{
	numberParameters = 2;
	
	range = parameters(0);
	variance = parameters(1);
	setDefaultTransforms();
}


ExponentialCF::~ExponentialCF()
{
}

inline double ExponentialCF::computeElement(const vec& A, const vec& B) const
{
	return calcExponential(A - B);
}

inline double ExponentialCF::computeDiagonalElement(const vec& A) const
{
	return calcExponentialDiag();
}

inline double ExponentialCF::calcExponential(const vec& V) const
{
	return variance * exp( -sqrt( arma::accu( arma::square(V) ) ) / (2.0*range));
}

inline double ExponentialCF::calcExponentialDiag() const
{
	return variance;
}

void ExponentialCF::setParameter(unsigned int parameterNumber, const double value)
{
	
	



	switch(parameterNumber)
	{
		case 0 : range = value;
					break;
		case 1 : variance = value;
					break;
		default: 
					break;
	}
}

double ExponentialCF::getParameter(unsigned int parameterNumber) const
{
	
	

	switch(parameterNumber)
	{
		case 0 : return(range);
					break;
		case 1 : return(variance);
					break;
		default: 
					break;
	}
	Rprintf("Warning: should not have reached here in GaussianCF::getParameter");
	return(0.0);
}

string ExponentialCF::getParameterName(unsigned int parameterNumber) const
{
	
	

	switch(parameterNumber)
	{
		case 0 : return("Range");
					break;
		case 1 : return("Variance");
					break;
		default: break;

	}
	return("Unknown parameter");
}

void ExponentialCF::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{
	
	

	Transform* t = getTransform(parameterNumber);
	double gradientModifier = t->gradientTransform(getParameter(parameterNumber));

	switch(parameterNumber)
	{
		case 0 :
		{
			mat DM;
			DM.set_size(PD.n_rows, PD.n_cols);
			computeSymmetric(PD, X);
			computeDistanceMatrix(DM, X);
			// elem_mult_inplace(0.5 * sqrt(DM)  * (gradientModifier / pow(range, 2.0)), PD);
            PD %= 0.5 * arma::sqrt(DM)  * (gradientModifier / pow(range, 2.0));
			return;
			break;
		}

		case 1 :
		{
			computeSymmetric(PD, X);
			PD *= (gradientModifier / variance);
			return;
			break;
		}
	}
	Rprintf("Warning: should not have reached here in GaussianCF::getParameterPartialDerivative");
}
