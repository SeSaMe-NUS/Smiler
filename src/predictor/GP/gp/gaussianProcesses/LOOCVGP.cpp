/*
 * LOOCV.cpp
 *
 *  Created on: Dec 27, 2013
 *      Author: zhouyiming
 */

#include "LOOCVGP.h"

using namespace std;
//using namespace itpp;//comment by jingbo
using namespace arma;

//(int Inputs, int Outputs, mat& Xdata, vec& ydata, CovarianceFunction& cf);
LOOCVGP::LOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata, vec& weight,
		CovarianceFunction& cf) :
		MetaGP(Inputs, Outputs, Xdata, ydata, cf), Weight(weight) {  //initialize base class
	//assert(Locations.rows() == Observations.size());//comment by jingbo
	assert(Locations.n_rows == Observations.size());
	///weightSum= sum(Weight);
}

//LOOCVGP::LOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata,
//		CovarianceFunction& cf) :
//		MetaGP(Inputs, Outputs, Xdata, ydata, cf) //initialize base class
//{
//	//assert(Locations.rows() == Observations.size());//comment by jingbo
//	assert(Locations.n_rows == Observations.size());
//	//add by jingbo
//
//	//vec weight = ones(Observations.size());
//	//weight = weight / Observations.size();
//	Weight = ones(Observations.size());
//	//weightSum = sum(Weight);
//}

LOOCVGP::~LOOCVGP() {
}

void LOOCVGP::computeMV(vec &Mean, vec &Variance) const {
	mat Sigma(Observations.size(), Observations.size());

	//covFunc.covariance(Sigma, Locations);//comment by jingbo
	covFunc.computeSymmetric(Sigma, Locations); // K = K(X,X)// add by jingbo

	mat invSigma = computeInverseFromCholesky(Sigma);

	vec alpha = invSigma * Observations;

	//Mean = Observations - elem_div(alpha, diag(invSigma));//comment by jingbo
	Mean = Observations - alpha / diagvec(invSigma); //comment by jingbo

	vec ONES = ones(Mean.size());
	//Variance = elem_div(ONES, diag(invSigma));//comment by jingbo

	Variance = ONES / diagvec(invSigma); //add by jingbo
}


double LOOCVGP::loglikelihood() const {
	vec tempMean, tempVar;
	computeMV(tempMean, tempVar);
	//	vec alpha = ls_solve(Sigma, Observations);
	vec out1 = -0.5 * log(tempVar);

	//vec out2 = 0.5 * elem_div(elem_mult(Observations - tempMean, Observations
	//		- tempMean), tempVar);//comment by jingbo

	vec sqrMean = square(Observations - tempMean); //add by jingbo

	vec out2 = -0.5 * (sqrMean / tempVar); //add by jingbo

	//return (out1 + out2) * Weight + 0.5 * log(2 * pi);//comment by jingbo
	vec sv = (out1 + out2 - 0.5 * log(2 * datum::pi)) % Weight; //comment by jingbo
	//we want to maximum the loglikelihood, therefore, this is negative loglikelihood
	//the objective is to find the minimum of negative loglikelihood
	return sum(sv);///weightSum; //comment by jingbo
}

double LOOCVGP::standardSquareError() const {
	vec tempMean, tempVar;
	computeMV(tempMean, tempVar);

	vec sqrMean = square(Observations - tempMean); //add by jingbo
	vec out2 = 0.5 * (sqrMean / tempVar); //add by jingbo

	return sum(out2);
}

/**
 *
 * @return
 */
double LOOCVGP::objective() const {
	//minimum the negative log likelihood
	return -1*loglikelihood();
}

vec LOOCVGP::gradient() const {
	//minimum the negative log likelihood
	return -1*getGradientVector();
}

vec LOOCVGP::getGradientVector() const {


	vec grads(covFunc.getNumberParameters());

	mat Sigma(Observations.size(), Observations.size());
	mat cholSigma(Observations.size(), Observations.size());

	//covFunc.covariance(Sigma, Locations);//comment by jingbo
	covFunc.computeSymmetric(Sigma, Locations); // K = K(X,X)//add by jingbo

	computeCholesky(Sigma, cholSigma);
	mat invSigma = computeInverseFromCholesky(Sigma);
	vec alpha = invSigma * Observations;

	//	vec alpha = ls_solve(Sigma, Observations);
	mat partialDeriv(Observations.size(), Observations.size());

	for (uint j = 0; j < covFunc.getNumberParameters(); j++) {
		//covFunc.covarianceGradient(partialDeriv, j, Locations);//comment by jingbo
		covFunc.getParameterPartialDerivative(partialDeriv, j, Locations); //add by jingbo;

		mat Zj = invSigma * partialDeriv;
		vec tempAl = Zj * alpha;
		mat tempZj = Zj * invSigma;
		grads(j) = 0.0;

		//for (uint i = 0; i < Observations.size(); i++) {
		//	grads(j) += Weight(i)*(alpha(i) * tempAl(i) - 0.5 * (1 + alpha(i) * alpha(i)
		//			/ invSigma(i, i)) * tempZj(i, i)) / invSigma(i, i);

		//}

		//use matrix operation
		grads(j) = sum(	Weight% (alpha % tempAl	- 0.5 * (1 + alpha % alpha / invSigma.diag()) % tempZj.diag()) / invSigma.diag());///weightSum;
		//grads(j) = -grads(j);//here!

		//grads(j) *= Weight(j);

	}
	return grads;
}

