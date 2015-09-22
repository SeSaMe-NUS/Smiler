/*
 * MetaGP.h
 *
 *  Created on: Jan 24, 2014
 *      Author: zhouyiming
 */

#ifndef METAGP_H_
#define METAGP_H_


#include "ForwardModel.h"
#include "../optimisation/Optimisable.h"
#include "../covarianceFunctions/CovarianceFunction.h"
#include "../covarianceFunctions/Transform.h"
#include "../psgp_common.h"



using namespace std;
using namespace arma;
using namespace psgp_arma;

class MetaGP : public ForwardModel, public Optimisable{

public:
	MetaGP(int Inputs, int Outputs, mat& Xdata, vec& ydata, CovarianceFunction& cf);

	virtual ~MetaGP();

	 void   makePredictions(vec& Mean, vec& Variance, const mat& Xpred, const mat& C) const;
	 void   makePredictions(vec& Mean, vec& Variance, const mat& Xpred, CovarianceFunction &cf) const;
	 void   makePredictions(vec& Mean, vec& Variance, const mat& Xpred) const;

	 void 	makeIterativePredictions( vec & _XtstMeanIterative, mat & _XtstVarIterative,
			 CovarianceFunction& cf, vec &  _nextStepMean, vec &  _nextStepVar);


	 vec    getParametersVector() const;
	 void   setParametersVector(const vec p);

	 void 	setXYtrn(mat& xtrn, vec &ytrn){
		 	 Locations = xtrn;
		 	 Observations = ytrn;
	 }
	 virtual void setWeight(vec& weight){
		 return;
	 }

	 void 	printMatrix(mat data) const;
	 void 	printVector(vec data) const;
	 void 	printCovPara(CovarianceFunction &cf) const;
	 void   estimateParameters();

public:
	 virtual double objective()const =0;
	 virtual vec    gradient()const =0;
	 virtual double standardSquareError() {return 0;}


protected:

	 void updateXtstMeanVar(int N, vec& prdMean, vec& prdVar, mat& invW, mat& EYE, vec& alpha, vec& q,
	 		vec& _XtstMeanItr, mat& _XtstVarItr);

	vec  computeAlpha(const mat& Sigma ) ;
	//mat    computeCholesky(const mat& iM) const;
	bool   computeCholesky(const mat& M, mat & cholFact) const;
	mat    computeInverseFromCholesky(const mat& C) const;
	bool   tryArmaChol(const mat& iM, mat & cholFactor) const;




	CovarianceFunction& covFunc;
	mat& Locations;
	vec& Observations;
};

#endif /* METAGP_H_ */
