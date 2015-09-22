/*
 * WLOOCVGP.h
 *
 *  Created on: Feb 10, 2014
 *      Author: zhouyiming
 */

#ifndef WLOOCVGP_H_
#define WLOOCVGP_H_

#include "LOOCVGP.h"

class WLOOCVGP: public LOOCVGP {
public:
	//WLOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata, CovarianceFunction& cf, vec& Xseed);
	WLOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata, vec& weight, CovarianceFunction& cf, vec& Xseed);
	virtual ~WLOOCVGP();

	//void estimateParameters();


public:
	 virtual double objective() const;
	 virtual vec    gradient() const;

	 vec    getGradientVector() const;
	 double loglikelihood() const;

private:
	 vec loglikelihoodVector() const;


private:
	 vec Xseed;
};

#endif /* WLOOCVGP_H_ */
