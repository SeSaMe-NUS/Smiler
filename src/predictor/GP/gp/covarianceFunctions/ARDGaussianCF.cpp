#include "ARDGaussianCF.h"


using namespace arma;

ARDGaussianCF::ARDGaussianCF(vec lengthscales, double var) : CovarianceFunction("automatic relevance determination (ARD) Gaussian")
{
	numberParameters = lengthscales.size()+1;
	setDefaultTransforms();

	ranges = lengthscales;
	variance = var;
}

ARDGaussianCF::ARDGaussianCF(double lengthscales, double var, int dims) : CovarianceFunction("automatic relevance determination (ARD) Gaussian")
{
	numberParameters = dims+1;
	setDefaultTransforms();

	ranges.set_size(dims);

	for(uint i=0;i<dims;i++){
		ranges[i]=lengthscales;
	}

	variance = var;
}

/*
ARDGaussianCF::ARDGaussianCF(vec parameters) : CovarianceFunction("Isotropic Gaussian")
{
	numberParameters = 2;
	assert(parameters.size() == getNumberParameters());
	ranges = parameters(0);
	variance = parameters(1);
	setDefaultTransforms();
}
*/


ARDGaussianCF::~ARDGaussianCF()
{
}

inline double ARDGaussianCF::computeElement(const vec& A, const vec& B) const{

	return calcGaussian(A - B);
}


inline double ARDGaussianCF::computeDiagonalElement(const vec& A) const{
	return calcGaussianDiag();
}


inline double ARDGaussianCF::calcGaussian(const vec& V) const
{
    //return variance * exp( -0.5 * sum(sqr(V)) / sqr(range) );//comment by jingbo
	/*cout << "range" << endl;
	cout << range << endl;*/


	double res = (variance*variance) * exp( -0.5 * sum(square(V/ranges)) );//add by jingbos
	return res;
}


inline double ARDGaussianCF::calcGaussianDiag() const
{
	return variance*variance;
}


void ARDGaussianCF::setParameter(unsigned int parameterNumber, const double value)
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	//set length scale
	if(parameterNumber<ranges.size()){
		ranges[parameterNumber]=value;
	}
	else{
		//set variance
		variance = value;
	}

}


vec ARDGaussianCF::getLenthScaleVector() const {

	vec lsv(ranges.size());
	for(int i=0;i<ranges.size();i++){
		lsv(i) = ranges(i);
	}
	return lsv;
}


vec ARDGaussianCF::getVarScaleVector() const{
	vec vsv(1);
	vsv(0) = variance;
	return vsv;
}

double ARDGaussianCF::getParameter(unsigned int parameterNumber) const
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	//set length scale
	if(parameterNumber<ranges.size()){
		return ranges[parameterNumber];
	}
	else{
		//set variance
		return variance;
	}

	cerr << "Warning: should not have reached here in GaussianCF::getParameter" << endl;
	return(0.0);
}

string ARDGaussianCF::getParameterName(unsigned int parameterNumber) const
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	//set length scale
	if(parameterNumber<ranges.size()){
		ostringstream ss;
		ss << parameterNumber;
		string res = "Range "+ ss.str();
		return res;
	}
	else{
		//set variance
		return "variance";
		}

	return("Unknown parameter");
}

void ARDGaussianCF::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	Transform* t = getTransform(parameterNumber);
	double gradientModifier = t->gradientTransform(getParameter(parameterNumber));


	if (parameterNumber < ranges.size()) {

		mat DM;
		//DM.set_size(PD.rows(), PD.cols());//comment by jingbo
		DM.set_size(PD.n_rows, PD.n_cols);//add by jingbo
		computeSymmetric(PD, X);
		computeDistanceMatrix(parameterNumber,DM, X);

		//elem_mult_inplace(DM  * (gradientModifier / pow(range, 3.0)), PD);//comment by jingbo,  B = elem_mult(A, B).
		//sqDist * correlation(sqDist) / pow(lengthScale,3.0);

		PD= ((DM  * (gradientModifier / pow(ranges[parameterNumber], 3.0)))) % PD;//add by jingbo

		return;

	}else{
		computeSymmetric(PD, X);
		PD *= (2*gradientModifier / variance);
		return;

	}

	cerr << "Warning: should not have reached here in GaussianCF::getParameterPartialDerivative" << endl;
}






//junk

/*
*
 * the partialDerivative of the Covariance between matrix X and a single input Xseed

void ARDGaussianCF::getParameterPartialDerivative(vec & pdv, const unsigned int parameterNumber, const mat & X, const vec & Xseed) const{
	assert(parameterNumber < getNumberParameters());
	assert(parameterNumber >= 0);

	Transform* t = getTransform(parameterNumber);
	double gradientModifier = t->gradientTransform(getParameter(parameterNumber));

	if(paramterNumber < ranges.size()){

		vec DV;
		DV.set_size(pdv.n_rows);
		computeCovariance(pdv,X,Xseed);
		computeDistanceVector(parameterNumber,DV,X,Xseed);

		pdv = ((DV  * (gradientModifier / pow(ranges[parameterNumber], 3.0)))) % pdv;

		return;

	}else{

		computeCovariance(pdv,X,Xseed);
		pdv *= (2*gradientModifier / variance);

		return;
	}

	cerr << "Warning: should not have reached here in GaussianCF::getParameterPartialDerivative" << endl;
}*/

