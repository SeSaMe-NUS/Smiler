/*
 * knnReg.cpp
 *
 *  Created on: Feb 17, 2014
 *      Author: zhouyiming
 */

#include "KnnReg.h"

//KnnReg::KnnReg(int Inputs, int Outputs,  mat& Xdata,  vec& ydata):
//		Locations(Xdata), Observations(ydata) {
//	// TODO Auto-generated constructor stub
//	Weight = ones(Observations.size());
//}
//
//KnnReg::KnnReg( mat& Xdata,  vec& ydata):Locations(Xdata), Observations(ydata){
//	// TODO Auto-generated constructor stub
//	Weight = ones(Observations.size());
//}
//
//KnnReg::KnnReg(int Inputs, int Outputs, mat& Xdata, vec& ydata, vec& weight):Locations(Xdata), Observations(ydata), Weight(weight){
//
//}
//KnnReg::KnnReg( mat &Xdata, vec & ydata, vec& weight):Locations(Xdata), Observations(ydata), Weight(weight){
//
//}

KnnReg::KnnReg(){

}

KnnReg::~KnnReg() {
	// TODO Auto-generated destructor stub
}


//void KnnReg::makePredictions( vec& _Mean, vec& _Variance) const{
//
//	_Mean(0) = sum(Observations%Weight)/sum(Weight);
//
//	_Variance(0) = sum(square(Observations-_Mean(0))%Weight)/sum(Weight)+0.0001;
//}

void KnnReg:: makePredictions( vec & ydata, vec & weight, vec& _Mean, vec& _Variance) const{



	vec iw=ones(weight.n_rows)/(weight+0.0001);

	_Mean(0) = sum(ydata%iw)/sum(iw);


	_Variance(0) = sum(square(ydata-_Mean(0))%iw)/sum(iw)+0.0001;

}


//void KnnReg:: makePredictions( vec & ydata, vec& _Mean, vec& _Variance) const{
//
//	_Mean(0) = sum(ydata)/ydata.n_rows;
//	_Variance(0) = sum(square(ydata-_Mean(0)))/ydata.n_rows+0.0001;
//
//}
