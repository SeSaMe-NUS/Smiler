#include "IdentityTransform.h"

IdentityTransform::IdentityTransform()
{
	transformName = "Not transformed";
}

IdentityTransform::~IdentityTransform()
{
}

double IdentityTransform::forwardTransform(const double a) const
{
	return a; 
}

double IdentityTransform::backwardTransform(const double b) const
{
	return b;
}

double IdentityTransform::gradientTransform(const double g) const
{
	return 1.0;
}
