#include "psgp_common.h"

using namespace arma;

namespace psgp_arma {

/**
 * Returns the lower triangular elements (including diagonal)
 * of a matrix, in row-major order.
 *
 * @param M An NxN square matrix
 * @return The vector of N(N+1)/2 lower triangular elements of M
 * @see ltr_mat, utr_vec, utr_mat
 */
vec ltr_vec(mat M)
{
    int N = M.n_cols;


    int k = 0;
    vec v(N*(N+1)/2);

    for (int i=0; i<N; i++)
        for (int j=0; j<=i; j++)
            v(k++) = M(i,j);
    return v;
}

/**
 * Returns the upper triangular elements (including diagonal)
 * of a matrix, in row-major order.
 *
 * @param M An NxN square matrix
 * @return The vector of N(N+1)/2 upper triangular elements of M
 * @see utr_mat, ltr_vec, ltr_mat
 */
vec utr_vec(mat M)
{
    int N = M.n_cols;


    int k = 0;
    vec v(N*(N+1)/2);

    for (int i=0; i<N; i++)
        for (int j=i; j<N; j++)
            v(k++) = M(i,j);

    return v;
}

/**
 * Returns a lower triangular matrix where the elements are taken
 * from the argument vector in row-major order.
 *
 * @param v A vector of N(N+1)/2 lower triangular elements
 * @return A NxN lower triangular matrix
 * @see ltr_vec, utr_vec, utr_mat
 */
mat ltr_mat(vec v)
{
    // Retrieve dimension of matrix
    int N =  (int) floor(sqrt(2.*v.n_elem));



    mat M = arma::zeros(N,N);
    int k = 0;

    for (int i=0; i<N; i++)
            for (int j=0; j<=i; j++)
                M(i,j) = v(k++);

    return M;
}

/**
 * Returns an upper triangular matrix where the elements are taken
 * from the argument vector in row-major order.
 *
 * @param v A vector of N(N+1)/2 upper triangular elements
 * @return A NxN upper triangular matrix
 * @see ltr_vec, utr_vec, utr_mat
 */
mat utr_mat(vec v)
{
    // Retrieve dimension of matrix
    int N =  (int) floor(sqrt(2.*v.n_elem));



    mat M = arma::zeros(N,N);
    int k = 0;

    for (int i=0; i<N; i++)
            for (int j=i; j<N; j++)
                M(i,j) = v(k++);

    return M;
}

/*
double cond(mat M, int p)
{

    return norm(M)*norm(inv(M));
}
*/

/**
 * Returns a random permutation of numbers between 0 and N-1
 */
uvec randperm(int n)
{


	// if (n == 1) return to_ivec(zeros(1));
	if (n == 1) return uvec().zeros(1);

    vec rndNums = arma::randu(n);
	return sort_index(rndNums);
}

/**
 * Returns the vector of minimum elements from 2 vectors, i.e.
 * z(i) = min(u(i), v(i)).
 */
vec min(vec u, vec v)
{


    vec z(u.n_elem);

    for (unsigned int i=0; i<u.n_elem; i++)
        z(i) = std::min(u(i), v(i));

    return z;
}

/**
 * Concatenation Z = [X y]
 */
mat concat_cols(mat X, vec y) {


    // mat Z(X.n_rows, X.n_cols+1);

    // for (int i=0; i<X.n_cols; i++) {
    	// Z.set_col(i, X.get_col(i));
    // }
    // Z.set_col(X.n_cols, y);

    // return Z;
    return arma::join_rows(X,y);
}

/**
 * Concatenation Z = [X Y]
 */
mat concat_cols(mat X, mat Y) {


    /*
    mat Z(X.n_rows, X.n_cols+Y.n_cols);

    for (int i=0; i<X.n_cols; i++) Z.set_col(i, X.get_col(i));
    for (int i=X.n_cols; i<X.n_cols+Y.n_cols; i++) Z.set_col(i, Y.get_col(i));

    return Z;
    */
    return arma::join_rows(X, Y);
}

/**
 * Empirical arithmetic mean along rows
 * Returns a column vector where each element is the mean of the corresponding row.
 */
vec mean_rows(mat X)
{
    // int D = X.n_cols;

    // vec m(D);
    // for (int i=0; i<D; i++) m(i) = mean(X.get_col(i));

    // return m;
	return arma::mean(X, COLUMN_ORDER);
}

/**
 * Empirical arithmetic mean along columns
 * Returns a row vector where each element is the mean of the corresponding column.
 */
vec mean_cols(mat X)
{
    /*
    int N = X.n_rows;

    vec m(N);
    for (int i=0; i<N; i++) m(i) = mean(X.get_row(i));
    return m;
    */
	return arma::mean(X, ROW_ORDER);
}


/**
 * Unbiased empirical covariance of the rows in X. The mean
 * of the rows is also return (for reuse).
 */
mat cov(mat X, vec &xmean)
{
    int N = X.n_rows;

    mat matxmean = mat(1,X.n_cols);
    matxmean.row(0) = mean_rows(X);

    mat Xcentred = X-repmat(matxmean, N, 1);

    mat C = 1.0/(N-1) * Xcentred.t() * Xcentred;

    xmean = matxmean.row(0);
    return C;
}

/** Covariance of the rows of a matrix X **/
mat cov(mat X)
{
    mat C;
    vec xmean;
    C = cov(X, xmean);
    return C;
}

/**
 * Normalises a data set comprising a set of inputs X and a set of outputs y.
 * The X and y arguments are overridden by their normalised versions.
 * The mean and covariance of the original dataset are also returned.
 */
void normalise(mat &X, vec &Xmean, vec &Xcovdiag)
{
    int N = X.n_rows;
    int D = X.n_cols;

    Xcovdiag  = diagvec(cov(X,Xmean));

    mat matXmean(1,D);

    matXmean.row(0) = Xmean;

    mat Xcentred = X - repmat(matXmean, N, 1);
    mat Xsphered(N,D);

    for (int i=0; i<D; i++)
        Xsphered.col(i) = 1.0/sqrt(Xcovdiag(i)) * Xcentred.col(i);

    X = Xsphered;
}

void normalise(mat &X)
{
    vec Xmean;
    vec Xcovdiag;
    normalise(X, Xmean, Xcovdiag);
}

void denormalise(mat &X, vec Xmean, vec Xcovdiag)
{
    int N = X.n_rows;
    int D = X.n_cols;

    mat matXmean(1,D);
    matXmean.row(0) = Xmean;

    // Rescale
    mat Xdesphered(N,D);
    for (int i=0; i<D; i++)
        Xdesphered.col(i) = sqrt(Xcovdiag(i))*X.col(i);

    // Add bias
    mat Xdecentred = Xdesphered + repmat(matXmean, N, 1);

    X = Xdecentred;
}

/** Return a matrix of zeros of indicated dimensions **/
mat psgp_zeros(int m, int n) {
	return mat().zeros(m,n);
}

/** Return a vector of zeros of indicated length **/
vec psgp_zeros(int m) {
	return vec().zeros(m);
}


/** Return a matrix of ones of indicated dimensions **/
mat ones(int m, int n) {
    return mat().ones(m,n);
}

/** Return a vector of ones of indicated length **/
vec ones(int m) {
	return vec().ones(m);
}

double sign(double x) {
    if (x<arma::math::eps()) {
        return 0.0;
    } else if (x<0.0) {
        return -1.0;
    } else {
        return 1.0;
    }
}

uvec sequence(int from, int to) {

	uvec v(to-from+1);
	for (int i=0; i<=to-from; i++) {
		v(i) = from+i;
	}
	return v;
}

int Rprintf(const char* format, ...) {
	//if (globalCheck == 0)
		//return 0;
	va_list vl;
	va_start(vl, format);
	vprintf(format, vl);
	va_end(vl);
}



} // END OF namespace psgp_arma




