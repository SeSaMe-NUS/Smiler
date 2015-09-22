/*
 * pgGaussianProcess.h
 *
 *  Created on: Jan 8, 2014
 *      Author: zhouyiming
 */

#ifndef PGGAUSSIANPROCESS_H_
#define PGGAUSSIANPROCESS_H_

#include "MMLGP.h"

class pgGaussianProcess : public MMLGP{
public:
	pgGaussianProcess(int Inputs, int Outputs, mat& Xdata, vec& ydata, CovarianceFunction& cf);
	virtual ~pgGaussianProcess();

	void makePredictions(vec &Mean, vec &Var, vec &mu, mat &sigma, CovarianceFunction &cf, int i);
	void printMatrix(mat data);
	mat inverse(mat &sigma, int index);
};

#endif /* PGGAUSSIANPROCESS_H_ */
