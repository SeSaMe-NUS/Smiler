#include "SumCovarianceFunction.h"

#include <iostream>
using namespace std;
using namespace psgp_arma;

// SumCovarianceFunction::SumCovarianceFunction(vector<CovarianceFunction> cfVec) : CovarianceFunction("Sum Covariance")
// {
// 	Rprintf("NOT IMPLEMENTED YET!!!");
// }

//add by jingbo
SumCovarianceFunction::SumCovarianceFunction() : CovarianceFunction("Sum Covariance")
{
	covFunctions.clear();
	numberParameters=0;
	setDefaultTransforms();
}

SumCovarianceFunction::SumCovarianceFunction(CovarianceFunction& cf) : CovarianceFunction("Sum Covariance")
{
	covFunctions.push_back(&cf);
	numberParameters = cf.getNumberParameters();
	setDefaultTransforms();
}

void SumCovarianceFunction::addCovarianceFunction(CovarianceFunction& cf)
{
	covFunctions.push_back(&cf);
	numberParameters = numberParameters + cf.getNumberParameters();
	setDefaultTransforms();
}

SumCovarianceFunction::~SumCovarianceFunction()
{
}

double SumCovarianceFunction::computeElement(const vec& A, const vec& B) const
{
	double k = 0.0;

	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		k = k + covFunctions[i]->computeElement(A, B);
	}

	return k;
}

double SumCovarianceFunction::computeDiagonalElement(const vec& A) const
{
	double k = 0.0;

	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		k = k + covFunctions[i]->computeDiagonalElement(A);
	}

	return k;
}

void SumCovarianceFunction::displayCovarianceParameters(int nspaces) const
{
    Rprintf("Covariance function : Sum\n");
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		Rprintf("+ Component: %d\n", i+1);
		covFunctions[i]->displayCovarianceParameters(nspaces+2);
	}
}

void SumCovarianceFunction::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{

	unsigned int pos = 0;

	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		for(unsigned int j = 0; j < (covFunctions[i]->getNumberParameters()) ; j++)
		{
			if(parameterNumber == pos)
			{
				covFunctions[i]->getParameterPartialDerivative(PD, j, X);
				return;
			}
			pos = pos + 1;
		}
	}
}


Transform* SumCovarianceFunction::getTransform(unsigned int parameterNumber) const
{
	unsigned int pos = 0;
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		for(unsigned int j = 0; j < (covFunctions[i]->getNumberParameters()) ; j++)
		{
			if(parameterNumber == pos)
			{
				return covFunctions[i]->getTransform(j);
			}
			pos = pos + 1;
		}
	}

	// We shouldn't reach here
	throw new std::exception();
}


void SumCovarianceFunction::setTransform(unsigned int parameterNumber, Transform* newTransform)
{
	
	

	//transforms[parameterNumber] = newTransform;

	unsigned int pos = 0;
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		for(unsigned int j = 0; j < (covFunctions[i]->getNumberParameters()) ; j++)
		{
			if(parameterNumber == pos)
			{
				covFunctions[i]->setTransform(j, newTransform);
				return;
			}
			pos = pos + 1;
		}
	}
}


void SumCovarianceFunction::setParameters(const vec p)
{
	int parFrom = 0;
	int parTo = 0;
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
	    // RB: Extract parameters for covariance function j
	    parFrom = parTo;
	    parTo += covFunctions[i]->getNumberParameters();

	    covFunctions[i]->setParameters( p.subvec(parFrom, parTo-1) ); 


	        
	    /*
	    for(int j = 0; j < (covFunctions[i]->getNumberParameters()) ; j++)
		{
			Transform* t = covFunctions[i]->getTransform(j);
			double d = t->backwardTransform(p(pos));
			covFunctions[i]->setParameter(j, d);
			pos = pos + 1;
		}
	    */
	}
}

vec SumCovarianceFunction::getParameters() const
{
	vec result;
	unsigned int pos = 0;

	result.set_size(getNumberParameters());

	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		for(unsigned int j = 0; j < (covFunctions[i]->getNumberParameters()) ; j++)
		{
			Transform* t = covFunctions[i]->getTransform(j);
			double d = t->forwardTransform(covFunctions[i]->getParameter(j));
			result[pos] = d;
			pos = pos + 1;
		}
	}
	return result;
}

void SumCovarianceFunction::setParameter(const unsigned int parameterNumber, const double value)
{

	Rprintf("SumCovarianceFunction::setParameter");

	int pnum = 0;
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		int numParams = covFunctions[i]->getNumberParameters();
		if(parameterNumber < (pnum + numParams))
		{
			covFunctions[i]->setParameter(parameterNumber - pnum, value);
			return;
		}
		pnum = pnum + numParams;
	}

	Rprintf("We shouldn't reach here - setParam : %d", parameterNumber);
}

double SumCovarianceFunction::getParameter(const unsigned int parameterNumber) const
{
	unsigned int pos = 0;
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		for(unsigned int j = 0; j < (covFunctions[i]->getNumberParameters()) ; j++)
		{
			if(parameterNumber == pos)
			{
				return covFunctions[i]->getParameter(j);
			}
			pos = pos + 1;
		}
	}
	
	return(0.0);

}

string SumCovarianceFunction::getParameterName(const unsigned int parameterNumber) const
{
	unsigned int pnum = 0;
	for(vector<CovarianceFunction *>::size_type i = 0; i < covFunctions.size(); i++)
	{
		unsigned int numParams = covFunctions[i]->getNumberParameters();
		if(parameterNumber < (pnum + numParams))
		{
			return (covFunctions[i]->getParameterName(parameterNumber - pnum));
		}
		pnum = pnum + numParams;
	}
	Rprintf("We shouldn't reach here - getParamName");
	return("Unknown");
}

