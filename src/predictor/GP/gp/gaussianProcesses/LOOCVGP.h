/*
 * LOOCV.h
 *
 *  Created on: Dec 27, 2013
 *      Author: zhouyiming
 */

#ifndef LOOCVGP_H_
#define LOOCVGP_H_

#include <assert.h>     /* assert */



//#include "itpp/itbase.h"//comment by jingbo

//using namespace itpp;//comment by jignbo
#include "MetaGP.h"
#include "../psgp_common.h" //add by jingbo

using namespace std;
using namespace arma;
using namespace psgp_arma;

class LOOCVGP : public MetaGP
{
public:
	LOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata,vec& weight, CovarianceFunction& cf );
	//LOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata, CovarianceFunction& cf);

 	virtual ~LOOCVGP();


	void   estimateParameters();

	virtual void setWeight(vec& weight){
		this->Weight = weight;
	}

public:
	//these two functions are for optimization
	 virtual double objective() const;
	 virtual vec    gradient() const;


	 vec    getGradientVector() const;
	 double loglikelihood() const;
	 double standardSquareError() const;

protected:

	void computeMV(vec &Mean, vec &Variance) const;

	vec& Weight;
	//double weightSum;

};

#endif /* LOOCV_H_ */
