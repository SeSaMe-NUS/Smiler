#include "Matern5CF.h"

/*
 * Constructor - pass in the length scale
 */
Matern5CF::Matern5CF(double ls, double var)
    : CovarianceFunction("Matern 5/2 covariance function")
{
    // Make sure length scale is positive 
    
    
    lengthScale = ls;   // Length scale
    variance = var;
    
    numberParameters = 2;
    setDefaultTransforms();
}


/*
 * Constructor - pass in a vector of parameters
 * (in this case [lengthscale, variance])
 */
Matern5CF::Matern5CF(vec parameters)
    : CovarianceFunction("Matern 5/2 covariance function")
{
    // Set number of parameters and check parameter vector has correct size
    numberParameters = 2;    
    

    // Set parameters
    
    lengthScale = parameters(0);
    variance = parameters(1);

    setDefaultTransforms();
}


/*
 * Destructor
 */
Matern5CF::~Matern5CF()
{
}


/**
 * Return the name of the parameter of specified index
 */
string Matern5CF::getParameterName(unsigned int parameterNumber) const
{
    
    

    switch (parameterNumber)
    {
    case 0: 
        return "Length scale";

    case 1:
        return "Variance";
    }
    return "Paramater name not found (out of bound)";
}


/**
 * Set given parameter to the specified value
 */
void Matern5CF::setParameter(unsigned int parameterNumber, const double value)
{
    
    

    switch(parameterNumber)
    {
        case 0 : 
            lengthScale = value;
            break;
            
        case 1 : 
            variance = value;
            break;
     }
}


/**
 * Return the parameter at index parameterNumber
 */
double Matern5CF::getParameter(unsigned int parameterNumber) const
{
    
    

    switch(parameterNumber)
    {
        case 0 : 
            return(lengthScale);
            break;
            
        case 1 : 
            return(variance);
            break;
    }
    return 0.0;
}


/**
 * Covariance between two points A and B
 */
inline double Matern5CF::computeElement(const vec& A, const vec& B) const
{
    if (accu(A==B) == A.n_elem) return computeDiagonalElement(A);
    
    double r = sqrt(5.0) * arma::norm(A-B,2) / lengthScale;
     
    return variance * ( 1.0 + r + pow(r,2.0)/3.0 ) * exp(-r);
}

/**
 * Auto-covariance
 */
inline double Matern5CF::computeDiagonalElement(const vec& A) const
{
    return variance;
}


/** 
 * Gradient of cov(X) w.r.t. given parameter number
 */
void Matern5CF::getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const
{
    
    

    Transform* t = getTransform(parameterNumber);
    double gradientModifier = t->gradientTransform(getParameter(parameterNumber));

    switch(parameterNumber)
    {
        case 0 :
        {
            mat R2(PD.n_rows, PD.n_cols);
            computeDistanceMatrix(R2, (sqrt(5.0) / lengthScale) * X);
            mat R = sqrt(R2); 
            PD = (gradientModifier * (variance/(3.0*lengthScale)) * (R2 %(1.0+R)))
                 % exp(-R);
            break;
        }

        case 1 :
        {
            computeSymmetric(PD, X);
            PD *= (gradientModifier / variance);
            break;
        }
    }

}

