/*
 * MetaGP.cpp
 *
 *  Created on: Jan 24, 2014
 *      Author: zhouyiming
 */

#include "MetaGP.h"

#include <assert.h>
#include <iostream>



MetaGP::MetaGP(int Inputs, int Outputs, mat& Xdata,	vec& ydata, CovarianceFunction& cf) :
		Locations(Xdata), Observations(ydata), covFunc(cf), ForwardModel(Inputs,Outputs) {

}



MetaGP::~MetaGP() {
}

void MetaGP::printMatrix(mat data) const{
	cout<<"print matix row:"<<data.n_rows<<" cols:"<<data.n_cols<<endl;
	for (uint i = 0; i < data.n_rows; i++) {
		for (uint j = 0; j < data.n_cols; j++)
			cout << " " << data(i, j);
		cout << endl;
	}
}

void MetaGP::printVector(vec data) const{
	cout<<"print vec  row:"<<data.n_rows<<endl;
	for(uint i=0;i<data.n_rows;i++){
		cout<<" "<<data(i);
	}
	cout<<endl;
}

void MetaGP::printCovPara(CovarianceFunction &cf) const{
	cout<<"parameter of cov:"<<endl;
	printVector(covFunc.getParameters());
}

void MetaGP::makePredictions(vec& Mean, vec& Variance,
		const mat& Xpred) const {
	makePredictions(Mean, Variance, Xpred, covFunc);
}



void MetaGP::makePredictions(vec& Mean, vec& Variance,
		const mat& Xpred, CovarianceFunction &cf) const {

	//re-write by jingbo, do  it by charpter2, pp 19

	mat Sigma(Observations.size(), Observations.size());

	mat Cpred(Locations.n_rows, Xpred.n_rows);

	cf.computeCovariance(Cpred, Locations, Xpred); // k_* = k(X,x*)
	covFunc.computeSymmetric(Sigma, Locations); // K = K(X,X)


	mat L(Sigma.n_rows,Sigma.n_rows);
	computeCholesky(Sigma, L);//L=cholesky(K), K=L^{t}*L, where L is upper matrix


	//K=Ll*Lu
	mat Lu = trimatu(L);     //upper
	mat Ll = trimatl(L.t());// lower
	//alpha = Lu\(Ll\y), set beta=Ll\y, i.e. Ll*beta=y
	vec beta = arma::solve(Ll,Observations);
	//then alpha = lu\beta, i.e. Lu*alpha = beta
	vec alpha = arma::solve(Lu, beta);



	Mean = Cpred.t() * alpha; // mu* = k' * K^{-1} * y, i.e. mu = k_*^{t}*alpha

	//let v = Ll\k_*, Ll*v = k_*
	mat v = arma::solve(Ll, Cpred);//add by jingbo,  v = K^{-1} * k_*, pp 19, eq. 2.26
	vec variancePred(Xpred.n_rows);

	cf.computeDiagonal(variancePred, Xpred); // k* = K(x*,x*)
	//Variance = variancePred - diagvec(v.t() * v); // ( k* - k'*K^{-1}*k ), pp19, eq.2.26
	Variance = variancePred - (v.t() * v); // ( k* - k'*K^{-1}*k ), pp19, eq.2.26

	//end of rewrite by jingbo
}


/**
 *
 * make iterative prediction, the error is propagated
 * _XtstMeanIterative: the iterative mean and variance will also updated and shifted after the prediction, therefore
 * _XtstMeanIterative and _XtstVarIterative are also output result
 * _XtstVarIterative:
 *
 *
 *_nextStepMean:
 *_nextStepVar :
 *  refer to PG[1]: Gaussian Processes Priors with Uncertain inputs applications to multiple-step ahead time series forecasting
 *  and PG[2]: Gaussian Processes: Prediction at a Noisy Input and Application to Iterative Multiple-Step Ahead Forecasting of Time Series
 *
 */
void MetaGP::makeIterativePredictions( vec & _XtstMeanItr, mat & _XtstVarItr,
		CovarianceFunction &cf, vec &  _nextStepMean, vec &  _nextStepVar){

	//double lengthScale = covFunc.getParameter(0);
	//double variance = covFunc.getParameter(1);

	vec lengthScaleVector = cf.getLenthScaleVector();
	vec varScaleVector = cf.getVarScaleVector();

	double variance = varScaleVector(0);
	variance = variance*variance;

	int D = _XtstMeanItr.size();
	int N = Observations.size();
	mat W = zeros(D, D);
	mat invW = zeros(D, D);
	mat EYE(D,D);
	EYE.eye();

	// All the same at the moment
	if (lengthScaleVector.size() == 1) {
		double lengthScale = lengthScaleVector(0);
		for (int i = 0; i < D; i++) {
			double wii = pow(lengthScale, 2);
			W(i, i) = wii;
			invW(i, i) = 1 / wii;
		}
	} else {
		for (int i = 0; i < D; i++) {
			double wii = pow(lengthScaleVector(i), 2);
			W(i, i) = wii;
			invW(i, i) = 1 / wii;
		}
	}


	mat Sigma(Observations.size(), Observations.size());

	covFunc.computeSymmetric(Sigma, Locations); // K = K(X,X)

	mat L(Sigma.n_rows, Sigma.n_cols);
	computeCholesky(Sigma, L);//L=cholesky(K), K=L^{t}*L, where L is upper matrix
	//K=Ll*Lu
	mat Lu = trimatu(L); //upper
	mat Ll = trimatl(L.t());// lower

	//compute invser Sigma by cholesky
	mat invChol = solve(Lu,	eye(Lu.n_rows, Lu.n_rows));
	mat invSigma = invChol * invChol.t();

	//alpha = Lu/(Ll/y), set beta=Ll\y, i.e. Ll*beta=y
	vec beta = arma::solve(Ll, Observations);
	//then alpha = lu\beta, i.e. Lu*alpha = beta
	vec alpha = arma::solve(Lu, beta);

	// Compute mu first
	vec q(N);
	mat temp =  inv(_XtstVarItr + W);

	double deter = (det(invW * _XtstVarItr + EYE));//
	assert(deter>=0);

	for (int i = 0; i < N; i++) {
		vec xi = Locations.row(i).t();
		vec diff = _XtstMeanItr - xi;
		q(i) = 1 / sqrt(deter) * variance
				* exp(-0.5 * dot(diff, (temp * diff)));
	}
	_nextStepMean(0)=sum(q % alpha);


	mat Cpred(Locations.n_rows, 1);
	cf.computeCovariance(Cpred, Locations, _XtstMeanItr.t()); // k_* = k(X,x*)
	mat v = arma::solve(Ll, Cpred);//add by jingbo,  v = K^{-1} * k_*, pp 19, eq. 2.26

	vec variancePred(1);
	cf.computeDiagonal(variancePred, _XtstMeanItr.t()); // k* = K(x*,x*)

	variancePred = variancePred - diagvec(v.t() * v); // ( k* - k'*K^{-1}*k ), pp19, eq.2.26


	double res = 0.0;
	// page 172
	double deter2 = (det(2 * invW * _XtstVarItr + EYE));//noted abs may be wrong
	assert(deter2>=0);
	mat temp2 = 2 * invW - inv(W / 2 + _XtstVarItr);
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++) {

			vec xi = Locations.row(i).t();
			vec xj = Locations.row(j).t();
			vec xb = (xi + xj) / 2;
			double cgi = variance * exp(-0.5 * dot(_XtstMeanItr - xi, invW * (_XtstMeanItr - xi)));
			double cgj = variance * exp(-0.5 * dot(_XtstMeanItr - xj, invW * (_XtstMeanItr - xj)));

			double cb = 1 / sqrt(deter2) * exp(0.5 * dot(_XtstMeanItr - xb, temp2 * (_XtstMeanItr
					- xb)));

			res += invSigma(i, j) * cgi * cgj * (1 - cb) + alpha(i) * alpha(j)
					* cgi * cgj * cb;
		}
	}

	_nextStepVar(0) = variancePred(0) + res - pow(_nextStepMean(0), 2);


	// update // page 176 of PG[2]
	updateXtstMeanVar(N,  _nextStepMean,  _nextStepVar,  invW,  EYE,  alpha,  q,
			 _XtstMeanItr,  _XtstVarItr);


}



void MetaGP::updateXtstMeanVar(int N, vec& prdMean, vec& prdVar, mat& invW, mat& EYE, vec& alpha, vec& q,
		vec& _XtstMeanItr, mat& _XtstVarItr){


		int D = _XtstMeanItr.size();
		// update
		// page 176
		vec covariance = zeros(D);
		for (int i = 0; i < N; i++) {
			vec xi = Locations.row(i).t();
			mat C = inv(EYE + _XtstVarItr*invW);
			covariance += alpha(i) * q(i) * (xi - C * xi);
		}

		for (int i = 0; i < D - 1; i++)
			_XtstMeanItr(i) = _XtstMeanItr(i + 1);
		_XtstMeanItr(D - 1) = prdMean(0);
		for (int i = 0; i < D - 1; i++)
			for (int j = 0; j < D - 1; j++)
				_XtstVarItr(i, j) = _XtstVarItr(i + 1, j + 1);
		_XtstVarItr(D - 1, D - 1) = prdVar(0);

		for (int i = 0; i < D - 1; i++) {
			_XtstVarItr(i, D - 1) = covariance(i + 1);
			_XtstVarItr(D - 1, i) = _XtstVarItr(i, D - 1);
		}
}

vec MetaGP::getParametersVector() const {
	return covFunc.getParameters();
}

void MetaGP::setParametersVector(const vec p) {
	covFunc.setParameters(p);
}

/**
 * the cholFact is upper matrix
 */
bool MetaGP::computeCholesky(const mat& M, mat &cholFact) const{

	 return tryArmaChol(M, cholFact);
}

/**
 * the returned matrix is upper matrix
 */
/*mat MetaGP::computeCholesky(const mat& M) const {
	//mat M = iM; // oops, was i inadvertantly writing to this?

	mat cholFactor(M.n_rows, M.n_cols);

	bool success = tryArmaChol(M, cholFactor);
	return cholFactor;
}*/

//add by jingbo
bool MetaGP::tryArmaChol(const mat& iM, mat & cholFactor) const {

	const double ampl = 1;
	const int maxAttempts = 10;


	int l = 0;
	bool success=false;

	try {
		success = arma::chol(cholFactor,iM);
	} catch (std::runtime_error &e) {
		cout<<"** Error: Cholesky decomposition failed."<<endl;
	}

	if(success) {return true;}

	mat M=iM;
	while(!success&&l<maxAttempts){
		double noiseFactor = abs(ampl * (trace(M) / double(M.n_rows)));
		M=M+ (noiseFactor * arma::eye(M.n_rows, M.n_rows));

		try {
			success = arma::chol(cholFactor,M);
		} catch (std::runtime_error &e) {
				cout<<"** Error: Cholesky decomposition of tryArmaChol() failed."<<endl;

		}
		l++;
	}


	//bool success = arma::chol(cholFactor,iM);

//	if (success) {
//		return true;
//	} else {
//
//		mat M = iM; // oops, was i inadvertantly writing to this?
//
//		double noiseFactor = abs(ampl * (trace(M) / double(M.n_rows)));
//		while (!success) {
//			M = M + (noiseFactor * arma::eye(M.n_rows, M.n_rows));
//
//			if (l > maxAttempts) {
//				Rprintf("Unable to compute cholesky decomposition");
//				break;
//			}
//			l++;
//			noiseFactor = noiseFactor * 10;
//			success = chol(M, cholFactor);
//		}
//		return false;
//		Rprintf(
//				"Matrix not positive definite.  After %d attempts, %f added to the diagonal",
//				l, noiseFactor);
//	}
	return false;

}


mat MetaGP::computeInverseFromCholesky(const mat& C) const {

	mat cholFactor(C.n_rows, C.n_cols);

	computeCholesky(C,cholFactor);

	mat invChol = solve(cholFactor,	arma::eye(cholFactor.n_rows, cholFactor.n_rows));
	return invChol * invChol.t();
}



vec MetaGP::computeAlpha(const mat& Sigma) {

	mat L(Sigma.n_rows, Sigma.n_cols);
	computeCholesky(Sigma, L); //L=cholesky(K), K=L^{t}*L, where L is upper matrix
	//K=Ll*Lu
	mat Lu = trimatu(L); //upper
	mat Ll = trimatl(L.t()); // lower

	//alpha = Lu/(Ll/y), set beta=Ll\y, i.e. Ll*beta=y
	vec beta = arma::solve(Ll, Observations);
	//then alpha = lu\beta, i.e. Lu*alpha = beta
	vec alpha = arma::solve(Lu, beta);

	return alpha;

}

