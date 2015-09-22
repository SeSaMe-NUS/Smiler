#ifndef MATERN5CF_H_
#define MATERN5CF_H_

#include "CovarianceFunction.h"

/**
 * Isotropic Matern covariance function with nu=5/2.
 * 
 * (C) 2009 Remi Barillec <r.barillec@aston.ac.uk>
 */ 
class Matern5CF : public CovarianceFunction
{
public:
    Matern5CF(double lengthScale, double var);
    Matern5CF(vec parameters);
	virtual ~Matern5CF();

    // Compute covariance element cov(A,B) 
    inline double computeElement(const vec& A, const vec& B) const;
    
    // Compute autocovariance element cov(A,A)
    inline double computeDiagonalElement(const vec& A) const;
    
    // Partial derivative of covariance function with respect to a given parameter
    void    getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const;
    
    // Set and get a given parameter
    void    setParameter(unsigned int parameterNumber, const double value);
    double  getParameter(unsigned int parameterNumber) const;

    // Returns the name of a given parameter
    string  getParameterName(unsigned int parameterNumber) const;

private:
//    static const double nu = 2.5;      // Nu parameter
    double lengthScale;                // Length scale parameter
    double variance;                   // Variance parameter

};

#endif /*MATERN5CF_H_*/
    
