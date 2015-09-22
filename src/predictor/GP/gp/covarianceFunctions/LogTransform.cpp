#include "LogTransform.h"

LogTransform::LogTransform()
{
	transformName = "Log";
}

LogTransform::~LogTransform()
{
}

double LogTransform::forwardTransform(const double a) const
{
	return log(a); 
}

double LogTransform::backwardTransform(const double b) const
{
	if(b < -MAXEXP)
	{
		return arma::math::eps();
	}
	else
	{
		if(b > MAXEXP)
		{
			return exp(MAXEXP);
		}
	}
	return exp(b);
}

double LogTransform::gradientTransform(const double g) const
{
	return g;
}
