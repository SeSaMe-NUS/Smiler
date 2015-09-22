/*
 * WLOOCVGP.cpp
 *
 *  Created on: Feb 10, 2014
 *      Author: zhouyiming
 */

#include "WLOOCVGP.h"


WLOOCVGP::WLOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata, vec& weight,
		CovarianceFunction& cf, vec &Xseed) :
	LOOCVGP(Inputs, Outputs, Xdata, ydata, weight, cf)//initialize base class
{
	this->Xseed = Xseed;


}

//WLOOCVGP::WLOOCVGP(int Inputs, int Outputs, mat& Xdata, vec& ydata,
//		CovarianceFunction& cf, vec &Xseed) :
//	LOOCVGP(Inputs, Outputs, Xdata, ydata, cf)//initialize base class
//{
//	this->Xseed = Xseed;
//
//}


WLOOCVGP::~WLOOCVGP() {
	// TODO Auto-generated destructor stub
}

/**
 * compute the negative LOO-CV log likelihood for each data point
 */
vec WLOOCVGP::loglikelihoodVector() const{
	vec tempMean, tempVar;
	computeMV(tempMean, tempVar);
	//	vec alpha = ls_solve(Sigma, Observations);
	vec out1 = -0.5 * log(tempVar);

	//vec out2 = 0.5 * elem_div(elem_mult(Observations - tempMean, Observations
	//		- tempMean), tempVar);//comment by jingbo

	vec sqrMean=square(Observations-tempMean);//add by jingbo
	vec out2 = -0.5 * (sqrMean/tempVar); //add by jingbo


	//return (out1 + out2) * Weight + 0.5 * log(2 * pi);//comment by jingbo
	vec loo = (out1 + out2 -0.5 * log(2 * datum::pi) ) %  Weight;//comment by jingbo
	//we want to maximum the loglikelihood, therefore, this is negative loglikelihood
	//the objective is to find the minimum of negative loglikelihood
	return loo;
}



double WLOOCVGP::loglikelihood() const {

	vec Cseed;
	Cseed.set_size(Locations.n_rows);
	covFunc.computeCovariance(Cseed, Locations, Xseed);

	vec lw = loglikelihoodVector();
	double res = sum(lw % Cseed)/sum(Cseed);

	return res;

}



vec WLOOCVGP::getGradientVector() const {
	//the objective is negative loglikelihood, so the grad should also be negative



	vec grads(covFunc.getNumberParameters());

	mat Sigma(Observations.size(), Observations.size());
	mat cholSigma(Observations.size(), Observations.size());

	//covFunc.covariance(Sigma, Locations);//comment by jingbo
	covFunc.computeSymmetric(Sigma, Locations); // K = K(X,X)//add by jingbo



	computeCholesky(Sigma, cholSigma);
	mat invSigma = computeInverseFromCholesky(Sigma);
	vec alpha = invSigma * Observations;

	//Cseed is the variance between X and single input Xseed
	vec Cseed, CseedSqr;
	Cseed.set_size(Observations.size());
	covFunc.computeCovariance(Cseed,Locations,Xseed);
	CseedSqr = Cseed%Cseed;

	vec loo = loglikelihoodVector();
	double lwn = sum(loo%Cseed);
	double lwd = sum(Cseed);
	double lwdSqr = lwd*lwd;

	//pX is the matrix of (Xseed + Location)
	mat pX = Locations;

	mat matseed(Xseed);
	pX.insert_rows(0,matseed.t());


	//pXDeriv is the partial derivation of pX
	mat pXDeriv(Observations.size()+1,Observations.size()+1);
	//CseedDeriv is the partial derivation of Cseed
	vec CseedDeriv;
	//partialDeriv is the partial derivation of sigma
	mat partialDeriv;


	for (uint j = 0; j < covFunc.getNumberParameters(); j++) {

		covFunc.getParameterPartialDerivative(pXDeriv, j, pX);//add by jingbo;
		//get the partial derivation on Locations data
		partialDeriv=pXDeriv.submat(1,1,pXDeriv.n_rows-1,pXDeriv.n_cols-1);
		//get the partial derivation on Location data and single seed x
		CseedDeriv = pXDeriv.col(0).rows(1,pXDeriv.n_rows-1);
		//cout<<"pXDrive rows:"<<pXDeriv.n_rows<<" pXDerive cols:"<<pXDeriv.n_cols<<endl;


		mat Zj = invSigma * partialDeriv;
		vec tempAl = Zj * alpha;
		mat tempZj = Zj * invSigma;
		grads(j) = 0.0;
		vec looDeriv_j(Observations.size());


		looDeriv_j = Weight%(alpha%tempAl-0.5*(1+alpha%alpha/invSigma.diag())%tempZj.diag())
				/invSigma.diag();


		double lwnDerive_j= sum(CseedDeriv%loo + Cseed%looDeriv_j);
		double lwdDerive_j = sum(CseedDeriv);

		grads(j) = (lwnDerive_j*lwd-lwn*lwdDerive_j)/lwdSqr;

	}
	return grads;


}



/**
 * the gradient of LOO without RHKS weight, only the gradient of LOO loglikelihood
 */
vec WLOOCVGP::gradient() const{
	return -1*getGradientVector();
}



/**
 *
 * @return
 */
double WLOOCVGP::objective() const {
	// the objective is negative loglikelihood
	//we use gradient descent to find mimum value
	//in this case, we find the minimum the negative loglikelihood, i.e. the maximum of loglikelihood
	return -1*loglikelihood();
}

