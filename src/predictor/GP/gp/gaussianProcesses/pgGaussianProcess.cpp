/*
 * pgGaussianProcess.cpp
 *
 *  Created on: Jan 8, 2014
 *      Author: zhouyiming
 */

#include "pgGaussianProcess.h"
#include "math.h"
#define _USE_MATH_DEFINES

pgGaussianProcess::pgGaussianProcess(int Inputs, int Outputs, mat& Xdata,
		vec& ydata, CovarianceFunction& cf) :
	MMLGP(Inputs, Outputs, Xdata, ydata, cf) {

}

pgGaussianProcess::~pgGaussianProcess() {
}

mat pgGaussianProcess::inverse(mat &sigma, int index) {
	int D = sigma.n_cols;
	mat ans = zeros(D, D);
	if (index > 0 && index < D) {
		mat submat = sigma.submat(D - index, D - index, D - 1, D - 1);
		mat submat_inv = inv(submat);
		for (int i = 0; i < index; i++)
			for (int j = 0; j < index; j++)
				ans(i + D - index, j + D - index) = submat_inv(i, j);
	} else if (index >= D) {
		ans = inv(sigma);
	}
	return ans;
}

void pgGaussianProcess::printMatrix(mat data) {
	for (uint i = 0; i < data.n_rows; i++) {
		for (uint j = 0; j < data.n_cols; j++)
			cout << " " << data(i, j);
		cout << endl;
	}
}

void pgGaussianProcess::makePredictions(vec &Mean, vec &Var, vec &mu,
		mat &sigma, CovarianceFunction &cf, int index) {

	double lengthScale = covFunc.getParameter(0);
	double variance = covFunc.getParameter(1);
	variance = variance*variance;
	//cout << "length scale and variance:" << endl;
	//cout << lengthScale << " " << variance << endl;
	int D = mu.size();
	int N = Observations.size();
	mat W = zeros(D, D);
	mat invW = zeros(D, D);
	//mat invsigma = inverse(sigma, index);
	mat EYE = zeros(D, D);
	// All the same at the moment
	for (int i = 0; i < D; i++) {
		W(i, i) = pow(lengthScale, 2);
		invW(i, i) = pow(1 / lengthScale, 2);
		EYE(i, i) = 1;
	}

	mat Sigma(Observations.size(), Observations.size());

	covFunc.computeSymmetric(Sigma, Locations); // K = K(X,X)


	mat L(Sigma.n_rows,Sigma.n_cols);
	computeCholesky(Sigma,L);//L=cholesky(K), K=L^{t}*L, where L is upper matrix
	//K=Ll*Lu
	mat Lu = trimatu(L); //upper
	mat Ll = trimatl(L.t());// lower

	//alpha = Lu/(Ll/y), set beta=Ll\y, i.e. Ll*beta=y
	vec beta = arma::solve(Ll, Observations);
	//then alpha = lu\beta, i.e. Lu*alpha = beta
	vec alpha = arma::solve(Lu, beta);

	// Compute mu first
	vec q(N);

	mat temp = inv(sigma + W);
	//cout << "sigma:" << endl;
	//printMatrix(sigma);
	double deter = det(invW * sigma + EYE);
	//cout << "deter: ";
	//cout << deter << endl;
	for (int i = 0; i < N; i++) {
		vec xi = Locations.row(i).t();
		vec diff = mu - xi;
		q(i) = 1 / sqrt(deter) * variance
				* exp(-0.5 * dot(diff, (temp * diff)));
	}

	Mean(index) = dot(q, alpha);

	mat Cpred(Locations.n_rows, 1);
	cf.computeCovariance(Cpred, Locations, mu.t()); // k_* = k(X,x*)
	mat v = arma::solve(Ll, Cpred);//add by jingbo,  v = K^{-1} * k_*, pp 19, eq. 2.26

	vec variancePred(1);
	cf.computeDiagonal(variancePred, mu.t()); // k* = K(x*,x*)

	variancePred = variancePred - diagvec(v.t() * v); // ( k* - k'*K^{-1}*k ), pp19, eq.2.26

	cout << "current variancePred: " << variancePred(0) << endl;
	mat invSigma = inv(Sigma);

	double res = 0.0;
	// page 172
	double deter2 = det(2 * invW * sigma + EYE);
	temp = 2 * invW - inv(W / 2 + sigma);
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++) {
			vec xi = Locations.row(i).t();
			vec xj = Locations.row(j).t();
			vec xb = (xi + xj) / 2;
			double cgi = variance * exp(-0.5 * dot(mu - xi, invW * (mu - xi)));
			double cgj = variance * exp(-0.5 * dot(mu - xj, invW * (mu - xj)));
			double cb = 1 / sqrt(deter2) * exp(0.5 * dot(mu - xb, temp * (mu
					- xb)));

			res += invSigma(i, j) * cgi * cgj * (1 - cb) + alpha(i) * alpha(j)
					* cgi * cgj * cb;
		}
	}

	Var(index) = variancePred(0) + res - pow(Mean(index), 2);

	cout << Mean(index) << " " << Var(index) << endl;

	// update
	// page 176
	vec covariance = zeros(D);
	for (int i = 0; i < N; i++) {
		vec xi = Locations.row(i).t();
		mat C = inv(EYE + invW * sigma);
		covariance += alpha(i) * q(i) * (xi - C * xi);
	}

	for (int i = 0; i < D - 1; i++)
		mu(i) = mu(i + 1);
	mu(D - 1) = Mean(index);
	for (int i = 0; i < D - 1; i++)
		for (int j = 0; j < D - 1; j++)
			sigma(i, j) = sigma(i + 1, j + 1);
	sigma(D - 1, D - 1) = Var(index);

	for (int i = 0; i < D - 1; i++) {
		sigma(i, D - 1) = covariance(i + 1);
		sigma(D - 1, i) = sigma(i, D - 1);
	}


}
