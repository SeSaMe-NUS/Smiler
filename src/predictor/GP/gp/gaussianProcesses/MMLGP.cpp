#include "MMLGP.h"
#include <iostream>
#include <assert.h>

MMLGP::MMLGP(int Inputs, int Outputs, mat& Xdata,
		vec& ydata, CovarianceFunction& cf) :
		MetaGP( Inputs,  Outputs,  Xdata,  ydata,  cf)//initialize base class
{
	

}

MMLGP::~MMLGP() {
}



double MMLGP::loglikelihood() const {

	mat Sigma(Observations.size(), Observations.size());
	mat cholSigma(Observations.size(), Observations.size());

	covFunc.computeSymmetric(Sigma, Locations);


	computeCholesky(Sigma,cholSigma);

	mat invSigma = computeInverseFromCholesky(Sigma);
	vec alpha = invSigma * Observations;

//	vec alpha = ls_solve(Sigma, Observations);

	double out1 = 0.5 * dot(Observations, alpha);

	double out2 = arma::accu(arma::log(arma::diagvec(cholSigma)));

	double res = (-out1 - out2 - 0.5 * Observations.n_elem * log(2 * arma::math::pi()));


	return res;

}


double MMLGP::objective() const {
	//minimue the negative likelihood

	return -1*loglikelihood();
}

vec MMLGP::gradient() const {
	//ngative the gradient
	//minimue the negative likelihood
	return -1*getGradientVector();
}




vec MMLGP::getGradientVector() const {

	vec grads(covFunc.getNumberParameters());

		mat Sigma(Observations.n_elem, Observations.n_elem);
		mat cholSigma(Observations.n_elem, Observations.n_elem);

		covFunc.computeSymmetric(Sigma, Locations);
		computeCholesky(Sigma,cholSigma);
		mat invSigma = computeInverseFromCholesky(Sigma);
		vec alpha = invSigma * Observations;

	//	vec alpha = ls_solve(Sigma, Observations);
		mat W = (alpha * alpha.t() - invSigma);//Gaussian Processes for Machine Learning, the MIT Press, 2006, pp114, eqn 5.9

		mat partialDeriv(Observations.size(), Observations.size());

		for (unsigned int i = 0; i < covFunc.getNumberParameters(); i++) {
			covFunc.getParameterPartialDerivative(partialDeriv, i, Locations);
			//grads(i) = sum(sum(elem_mult(W, partialDeriv))) / 2;
			grads(i) = arma::accu(W % partialDeriv) / 2.0;
	// official - but slower		grads(i) = sum(diag(W * partialDeriv)) / 2;

		}

		return grads;

}


