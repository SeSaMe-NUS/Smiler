#include "GaussianCF.h"

using namespace arma;

GaussianCF::GaussianCF(double lengthscale, double var) : CovarianceFunction("Isotropic Gaussian")
{
	numberParameters = 2;
	setDefaultTransforms();

	range = lengthscale;
	variance = var;

}

GaussianCF::GaussianCF(vec parameters) : CovarianceFunction("Isotropic Gaussian")
{
	numberParameters = 2;
	assert(parameters.size() == getNumberParameters());
	range = parameters(0);
	variance = parameters(1);
	setDefaultTransforms();
}


GaussianCF::~GaussianCF()
{
}

inline double GaussianCF::computeElement(const vec& A, const vec& B) const
{
	return calcGaussian(A - B);
}

inline double GaussianCF::computeDiagonalElement(const vec& A) const
{
	return calcGaussianDiag();
}

inline double GaussianCF::calcGaussian(const vec& V) const
{
    //return variance * exp( -0.5 * sum(sqr(V)) / sqr(range) );//comment by jingbo
	/*cout << "range" << endl;
	cout << range << endl;*/
	double res = (variance *variance)* exp( -0.5 * sum(square(V))/(range*range) );//add by jingbos


	return res;
}

inline double GaussianCF::calcGaussianDiag() const
{
	return variance*variance;
}

void GaussianCF::setParameter(unsigned int parameterNumber, const double value)
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	switch(parameterNumber)
	{
		case 0 : range = value;
					break;
		case 1 : variance = value;
					break;
		default: assert(false);
					break;
	}
}

vec GaussianCF::getLenthScaleVector() const {
	vec lsv(1);
	lsv(0) = range;
	return lsv;
}

vec GaussianCF::getVarScaleVector() const{
	vec vsv(1);
	vsv(0) = variance;
	return vsv;
}

double GaussianCF::getParameter(unsigned int parameterNumber) const
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	switch(parameterNumber)
	{
		case 0 : return(range);
					break;
		case 1 : return(variance);
					break;
		default: assert(false);
					break;
	}
	cerr << "Warning: should not have reached here in GaussianCF::getParameter" << endl;
	return(0.0);
}

string GaussianCF::getParameterName(unsigned int parameterNumber) const
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

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

void GaussianCF::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	Transform* t = getTransform(parameterNumber);
	double gradientModifier = t->gradientTransform(getParameter(parameterNumber));

	switch(parameterNumber)
	{
		case 0 :
		{
			mat DM;
			//DM.set_size(PD.rows(), PD.cols());//comment by jingbo
			DM.set_size(PD.n_rows, PD.n_cols);//add by jingbo
			computeSymmetric(PD, X);
			computeDistanceMatrix(DM, X);

			//elem_mult_inplace(DM  * (gradientModifier / pow(range, 3.0)), PD);//comment by jingbo,  B = elem_mult(A, B).
			//sqDist * correlation(sqDist) / pow(lengthScale,3.0);

			PD= ((DM  * (gradientModifier / pow(range, 3.0)))) % PD;//add by jingbo

			return;
			break;
		}

		case 1 :
		{
			computeSymmetric(PD, X);
			PD *= (2*gradientModifier / variance);
			return;
			break;
		}
	}
	cerr << "Warning: should not have reached here in GaussianCF::getParameterPartialDerivative" << endl;
}
