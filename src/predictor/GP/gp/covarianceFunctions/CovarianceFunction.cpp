#include "CovarianceFunction.h"

#include <iostream>
using namespace std;
using namespace arma;
using namespace psgp_arma;

CovarianceFunction::CovarianceFunction(string name)
{
	covarianceName = name;
	transformsApplied = false;
	numberParameters = 0;
}

CovarianceFunction::~CovarianceFunction()
{
	for(int i=0;i<transforms.size();i++){
		delete transforms[i];
	}
	transforms.clear();
}


/**
 * Compute auto-covariance of a single input X  
 */ 
void CovarianceFunction::computeSymmetric(double &c, const vec& x) const 
{
   mat C(1,1);
   computeSymmetric(C,x.t());
   c = C(0,0);
}


void CovarianceFunction::computeSymmetric(mat& C, const mat& X) const
{
	// ensure that data dimensions match supplied covariance matrix
	
	

	if (X.n_rows == 1)
	{
	    // C.set(0, 0, computeDiagonalElement(X.row(0)));
        C(0,0) = computeDiagonalElement(X.row(0).t());
        return;
	}
	
	// calculate the lower and upper triangles
	double d;
	
	for(unsigned int i=0; i<X.n_rows ; i++)
	{
		for(unsigned int j=0; j<i; j++)
		{
		    d = computeElement(X.row(i).t(), X.row(j).t());
		    C(i, j) = d;
			C(j, i) = d;
		}
	}

	// calculate the diagonal part
	for(unsigned int i=0; i<X.n_rows ; i++)
	{
		C(i, i) = computeDiagonalElement(X.row(i).t());
	}
}

void CovarianceFunction::computeSymmetricGrad(vec& V, const mat& X) const
{

}


/**
 * Compute covariance between a set of inputs X and single input x
 */
void CovarianceFunction::computeCovariance(vec& c, const mat& X, const vec& x) const
{
    mat Xmat(x);
    mat C(X.n_rows,1);
    computeCovariance(C,X,Xmat.t());
    c = C.col(0);
}

/**
 *  Covariance between points (rows) in X1 and points in X2
 */
void CovarianceFunction::computeCovariance(mat& C, const mat& X1, const mat& X2) const
{
	
	

	for(unsigned int i=0; i<X1.n_rows ; i++)
	{
		for(unsigned int j=0; j<X2.n_rows; j++)
		{
			C(i, j) = computeElement(X1.row(i).t(), X2.row(j).t());
		}
	}
}

void CovarianceFunction::computeDiagonal(mat& C, const mat& X) const
{
	// calculate the diagonal part
	for(unsigned int i=0; i<X.n_rows ; i++)
	{
		C(i, i) = computeDiagonalElement(X.row(i));
	}
}

void CovarianceFunction::computeDiagonal(vec& C, const mat& X) const
{
	// calculate the diagonal part
	for(unsigned int i=0; i<X.n_rows ; i++)
	{
		C(i) = computeDiagonalElement(X.row(i).t());
	}
}

/**
 * Display information about current covariance function
 * If wanted, the output can be indented by a number space characters
 * as specified in argument.
 */
void CovarianceFunction::displayCovarianceParameters(int nspaces) const
{
	string space = string(nspaces, ' ');
	
	Rprintf("%s Covariance function : %s\n", space.c_str(), covarianceName.c_str());

	vec t = getParameters();

	for(unsigned int i=0; i < (t.size()); i++)
	{
		Rprintf("%s %s  (P%d) :", space.c_str(), getParameterName(i).c_str(), i);
		Rprintf("%f", transforms[i]->backwardTransform(t(i)));
		/*
		if(transformsApplied)
		{
			Rprintf(" (%s)", transforms[i]->type().c_str());
		}
		else
		{
            Rprintf( " (no transform applied)" );
		}
		*/
		Rprintf("\n");
	}

}

void CovarianceFunction::setParameters(const vec p)
{
	
	for(unsigned int i = 0; i < getNumberParameters() ; i++)
	{
		setParameter(i, transforms[i]->backwardTransform(p(i)));
	}
}

vec CovarianceFunction::getParameters() const
{
	
	vec result;
	result.set_size(getNumberParameters());
	for(unsigned int i = 0; i < getNumberParameters() ; i++)
	{
		result[i] = transforms[i]->forwardTransform(getParameter(i));
	}
	return result;
}

unsigned int CovarianceFunction::getNumberParameters() const
{
	return numberParameters;
}

void CovarianceFunction::setTransform(unsigned int parameterNumber, Transform* newTransform)
{
	transforms[parameterNumber] = newTransform;
}

void CovarianceFunction::setDefaultTransforms()
{
	int sz = getNumberParameters();


	for(int i = transforms.size(); i < sz; i++)
	{
		// LogTransform* defaultTransform = new LogTransform();
		IdentityTransform* defaultTransform = new IdentityTransform();

		transforms.push_back(defaultTransform);
	}
	transformsApplied = true;
}

Transform* CovarianceFunction::getTransform(unsigned int parameterNumber) const
{
	
	
	return transforms[parameterNumber];
}

/**
 * compute the distance per dimensions
 */
void CovarianceFunction::computeDistanceMatrix(uint dim_i, mat& DM, const mat& X) const{

	double value;

	for (unsigned int i = 0; i < X.n_rows; i++) {
		for (unsigned int j = 0; j < i; j++) {
			value = arma::accu(arma::square(X.row(i).col(dim_i) - X.row(j).col(dim_i)));
			DM(i, j) = value;
			DM(j, i) = value;
		}
		DM(i, i) = 0.0;
	}
}

/**
 * compute the distance per dimension between mat X and one single input Xseed
 */
void CovarianceFunction::computeDistanceVector(uint dim_i, vec& DV, const mat& X, const vec& Xseed) const {

	double value;

	for(uint i=0;i<X.n_rows;i++){

		value =arma::accu(arma::square(X.row(i).col(dim_i) - Xseed(dim_i)));
		DV(i)=value;

	}

}

void CovarianceFunction::computeDistanceMatrix(mat& DM, const mat& X) const
{
	double value;
	
	

	for(unsigned int i=0; i<X.n_rows ; i++)
	{
		for(unsigned int j=0; j<i; j++)
		{
			value = arma::accu( arma::square( X.row(i) - X.row(j) ) );
			DM(i, j) = value;
			DM(j, i) = value;
		}
		DM(i, i) = 0.0;
	}
}
