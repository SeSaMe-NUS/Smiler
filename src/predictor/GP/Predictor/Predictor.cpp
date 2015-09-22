/*



 * Predictor.cpp
 *
 *  Created on: Apr 2, 2014
 *      Author: zhoujingbo


#include "Predictor.h"



Predictor::~Predictor()
{
	// TODO Auto-generated destructor stub
}

/
*
 * _meanRes:
 * _varRes:
 * step:step-ahead prediction result

void Predictor<T>::getPredictonResult(int step, double & _meanRes, double & _varRes){
	double C=0;
	double meanSum=0;
	double varSum = 0;
	for(int i=0;i<numL;i++){

		for(int j=0;j<numK;j++){
			double cij = weightPrdMat[i][j];
			C+=cij;
			//the prediction result of each expert is stored in mean and var
			assert(step<mean[i][j].n_rows&&step<var[i][j].n_rows);

			meanSum = cij*(mean[i][j](step)) +meanSum;
			varSum = cij*(var[i][j](step)) + varSum;

		}
	}

	_meanRes = meanSum/C;
	_varRes = varSum/C;
}


void Predictor<T>::getPredictonResult(double & _meanRes, double & _varRes){
	 getPredictonResult(0, _meanRes,  _varRes);
}


// shift and concatenate: first delete the first column of Xtrn, and then add the Ytrn as the last column of Xtrn

void Predictor<T>::shiftConcatXtrn(){
	for(int i=0;i<numL;i++){
		mat& XtrnSlice = Xtrn[i];
		vec YtrnSlice = Ytrn[i];//NOTE: this is not reference since the Ytrn is inserted into the mat, and Ytrn will be updated later
		XtrnSlice.shed_col(0);
		XtrnSlice.insert_cols(XtrnSlice.n_cols,YtrnSlice);
	}
}


void Predictor<T>::initialParameter(int numL, int numK){

	mean.resize(numL);
	var.resize(numL);
	weightPrdMat.resize(numL);
	XtstMeanItr.resize(numL);
	XtstVarItr.resize(numL);

	iterativeSteps=0;


	for (int i = 0; i < numL; i++) {
		mean[i].resize(numK);
		var[i].resize(numK);
		XtstMeanItr[i].resize(numK);
		XtstVarItr[i].resize(numK);
		weightPrdMat[i].resize(numK);
		//initialize the item in the matrix
		for(int j=0;j<numK;j++){
			mean[i][j] = vec().zeros(0);
			var[i][j] = vec().zeros(0);
			weightPrdMat[i][j] = 1.;
		}
	}


}


*/

