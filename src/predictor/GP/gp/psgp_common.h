#ifndef ITPPEXT_H_
#define ITPPEXT_H_

// Use BLAS and LAPACK
#define ARMA_USE_BLAS
#define ARMA_USE_LAPACK

/**
 * Various generic functions required by PSGP
 */
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>     /* va_list, va_start, va_copy, va_arg, va_end */
//#include <RcppArmadillo.h>//comment by jingbo
#include "armadillo"//add by jingbo



#define COLUMN_ORDER 0
#define ROW_ORDER 1


//typedef  arma::mat mat;
//typedef arma::vec vec;
//typedef arma::uvec uvec;




namespace psgp_arma
{




arma::vec ltr_vec(arma::mat M);     // Vector of lower triangular elements
arma::vec utr_vec(arma::mat M);     // Vector of upper triangular elements
arma::mat ltr_mat(arma::vec v);     // Lower triangular matrix
arma::mat utr_mat(arma::vec v);     // Upper triangular matrix

double cond(arma::mat M, int p=2); // Condition number for matrix p-norm (1 or 2)

arma::uvec randperm(int n);  // Random permutation of numbers between 0 and N-1
arma::uvec sequence(int from, int to);

arma::vec min(arma::vec u, arma::vec v); // Minimum elements from 2 vectors of equal length

arma::mat concat_cols(arma::mat X, arma::vec y); // Concatenate matrix and vector
arma::mat concat_cols(arma::mat X, arma::mat Y); // Concatenate matrix and matrix

arma::vec mean_rows(arma::mat X);  // vector of column means
arma::vec mean_cols(arma::mat X);  // vector of row means
arma::mat cov(arma::mat X, arma::vec &xmean); // covariance of rows, also returns the mean
arma::mat cov(arma::mat X); // covariance of rows

void normalise(arma::mat &X);
void normalise(arma::mat &X, arma::vec &mean, arma::vec &covdiag);
void denormalise(arma::mat &X, arma::vec mean, arma::vec covdiag);

arma::mat psgp_zeros(int m, int n);
arma::mat ones(int m, int n);
arma::vec psgp_zeros(int m);
arma::vec ones(int m);

double norm();
double sign(double x);

int Rprintf(const char* format, ...);


} // END OF namespace armaEXT


#endif /*ITPPEXT_H_*/
