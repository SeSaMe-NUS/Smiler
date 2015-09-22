#include "GaussianLikelihood.h"

#include <cmath>

using namespace std;

GaussianLikelihood::GaussianLikelihood(const double lp)
{
	 likelihoodParameter = lp;
}

GaussianLikelihood::~GaussianLikelihood()
{
}


double GaussianLikelihood::updateCoefficients(double& q, double& r, const double Observation, 
											  const double ModelMean, const double ModelVariance) const
{
    // Lehel: Sec. 2.4
	double sigX2 = ModelVariance + likelihoodParameter;
	double logLik;
	r = - 1.0 / sigX2;
	q = - r * (Observation - ModelMean);
	
	logLik = - 0.5 * ( log(2.0 * M_PI * sigX2) + (Observation - ModelMean) * q );
	return logLik;
}
