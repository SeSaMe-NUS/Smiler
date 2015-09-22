/*
 * knnReg.h
 *
 *  Created on: Feb 17, 2014
 *      Author: zhouyiming
 */

#ifndef KNNREG_H_
#define KNNREG_H_

#include "../psgp_common.h" //add by jingbo

using namespace std;
using namespace arma;
using namespace psgp_arma;


class KnnReg {
public:
	//KnnReg(int Inputs, int Outputs, mat& Xdata, vec& ydata);
	//KnnReg(int Inputs, int Outputs, mat& Xdata, vec& ydata, vec& weight);
	//KnnReg( mat &Xdata, vec & ydata, vec& weight);
	//KnnReg( mat &Xdata, vec & ydata);
	KnnReg();
	virtual ~KnnReg();


	//void makePredictions( vec& _Mean, vec& _Variance) const;

	void makePredictions( vec & ydata, vec& weight, vec& _Mean, vec& _Variance) const;

	//void makePredictions( vec & ydata, vec& _Mean, vec& _Variance) const;

	 //mat& Locations;
	 //vec& Observations;
	 //vec& Weight;
};

#endif /* KNNREG_H_ */
