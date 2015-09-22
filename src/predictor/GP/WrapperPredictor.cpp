/*
 * WrapperSemiLazyPredictor.cpp
 *
 *  Created on: Apr 4, 2014
 *      Author: zhoujingbo
 */

#include "WrapperPredictor.h"

#include "../../searcher/Scan/GPUScan.h"
#include "../../searcher/WrapperScan.h"
#include "../../tools/DataOperator/DataProcess.h"
#include "Predictor/RegPredictor.h"
#include "Predictor/LOOCVPredictor.h"

#include <iostream>
#include <cmath>
using namespace std;

WrapperPredictor::WrapperPredictor() {
	// TODO Auto-generated constructor stub

}

WrapperPredictor::~WrapperPredictor() {
	// TODO Auto-generated destructor stub
}

void WrapperPredictor::initQuery(const char* queryFile,
		vector<vector<int> > & _qdata) {
	//load query
	DataProcess dp;
	dp.ReadFileInt(queryFile, _qdata);
}



/**
 * expending the Lvec and Kvec, the sequence is
 * Lvec: L/8, L/4, L/2, L
 * Kvec: K/8, K/4, K/2, K
 */
void WrapperPredictor::initLK(const int L, const int K, vector<int>& Lvec, vector<int>& Kvec){

	int lenLK = 4;

	Lvec.clear();
	Lvec.resize(4);

	Kvec.clear();
	Kvec.resize(4);

	for(int i=0;i<lenLK;i++){
		Lvec[i] = L/pow(2,lenLK-i-1);
		Kvec[i] = K/pow(2,lenLK-i-1);
	}

}


void WrapperPredictor::InitTSOneStepByScan(string inputFilename,
		string queryFile, int columnAv, int dim, int topk,
		BladeLoader<int>& _bldLoader, vector<int>& _Lvec, vector<int>& _Kvec,
		vector<vector<int> >& _resIdx, vector<vector<float> >& _dist,
		vector<vector<vector<int> > >& _Xtrn, vector<vector<int> >& _Ytrn,
		vector<vector<int> >& _qdata) {

	_bldLoader.loadDataInt(inputFilename.c_str(), columnAv);
	cout << "load data item:" << _bldLoader.data.size() << endl;

	initQuery(queryFile.c_str(), _qdata);

	RtrDTWScan(_bldLoader.data, _qdata, dim, topk, _resIdx, _dist);

	vector<vector<int> > XtrnSlice;
	vector<int> YtrnSlice;
	int tq_num = 1; //_qdata.size();
	int y_offset = 0;
	for (int i = 0; i < tq_num; i++) {
		_bldLoader.retrieveXYtrn(_resIdx[i], dim,y_offset, XtrnSlice, YtrnSlice);
	}

	cout << "XtrnSlice size:" << XtrnSlice.size() << " YtrnSlice size:"
			<< YtrnSlice.size() << endl;

	_Lvec.push_back(dim);
	_Kvec.push_back(topk);

	_Xtrn.push_back(XtrnSlice);
	_Ytrn.push_back(YtrnSlice);

}


void WrapperPredictor::InitTSOneStepByScan(string inputFilename,
		string queryFile, int columnAv, int dim, int topk, vector<int>& _Lvec,
		vector<int>& _Kvec, vector<vector<float> >& _dist,
		vector<vector<vector<int> > >& _Xtrn, vector<vector<int> >& _Ytrn,
		vector<vector<int> >& _qdata) {

	BladeLoader<int> _bldLoader;
	vector<vector<int> > _resIdx;

	InitTSOneStepByScan(inputFilename, queryFile, columnAv, dim, topk,
			_bldLoader, _Lvec, _Kvec, _resIdx, _dist, _Xtrn, _Ytrn, _qdata);

}

void WrapperPredictor::TSOneStepRegPredictor(string inputFilename,
		string queryFile, int columnAv, int dim, int topk, bool weighted) {

	vector<int> Lvec;
	vector<int> Kvec;

	vector<vector<vector<int> > > Xtrn;
	vector<vector<int> > Ytrn;
	vector<vector<float> > dist;
	vector<vector<int> > qdata;

	InitTSOneStepByScan(inputFilename, queryFile, columnAv, dim, topk,
			Lvec, Kvec, dist, Xtrn, Ytrn, qdata);

	RegPredictor<int>* regPrd = new RegPredictor<int>(Lvec, Kvec);
	if(weighted){
		regPrd->makePrediction(qdata[0],  Xtrn, Ytrn, dist);
	}else{
		regPrd->makePrediction(qdata[0], Xtrn, Ytrn);
	}

	double mean, var;
	regPrd->getPredictonResult(mean, var);

	cout << "mean:" << mean << " var:" << var << endl;
	delete regPrd;

}

void WrapperPredictor::TSOneStepLOOCVPredictor(string inputFilename,
		string queryFile, int columnAv, int dim, int topk, bool weighted) {

	vector<int> Lvec;
	vector<int> Kvec;

	vector<vector<vector<int> > > Xtrn;
	vector<vector<int> > Ytrn;
	vector<vector<float> > dist;
	vector<vector<int> > qdata;

	InitTSOneStepByScan(inputFilename, queryFile, columnAv, dim, topk,
			Lvec, Kvec, dist, Xtrn, Ytrn, qdata);


	LOOCVPredictor<int>* looPrd;


	looPrd = new LOOCVPredictor<int>(Lvec, Kvec, 5, 1, 1);

	if(weighted){
		looPrd->makePrediction(qdata[0], Xtrn, Ytrn, dist);
	}else{
		looPrd->makePrediction(qdata[0], Xtrn, Ytrn);
	}

	double mean, var;
	looPrd->getPredictonResult(mean, var);

	cout << "mean:" << mean << " var:" << var << endl;

	delete looPrd;

}



void WrapperPredictor::TSMulStepsRegPredictor(string inputFilename,
		string queryFile, int columnAv, int dim, int topk, int steps,bool weighted) {
	BladeLoader<int> bldLoader;
	vector<int> Lvec;
	vector<int> Kvec;

	vector<vector<int> > resIdx;
	vector<vector<float> > dist;
	vector<vector<vector<int> > > Xtrn;
	vector<vector<int> > Ytrn;
	vector<vector<int> > qdata;

	InitTSOneStepByScan(inputFilename, queryFile, columnAv, dim, topk,
			bldLoader, Lvec, Kvec, resIdx, dist, Xtrn, Ytrn, qdata);

	vector<vector<int> > resIdxQueryOne;
	resIdxQueryOne.clear();
	resIdxQueryOne.push_back(resIdx[0]);

	RegPredictor<int> *regPrd=new RegPredictor<int>(Lvec, Kvec);
	if(weighted){
	regPrd->makeItrPrediction(qdata[0],dist,bldLoader,resIdxQueryOne,14);
	}else{
		regPrd->makeItrPrediction(qdata[0],bldLoader,resIdxQueryOne,14);
	}

	delete regPrd;
}


void WrapperPredictor::TSMulStepsLOOCVPredictor(string inputFilename,
		string queryFile, int columnAv, int dim, int topk, int steps,
		bool weighted) {

	BladeLoader<int> bldLoader;
	vector<int> Lvec;
	vector<int> Kvec;

	vector<vector<int> > resIdx;
	vector<vector<float> > dist;
	vector<vector<vector<int> > > Xtrn;
	vector<vector<int> > Ytrn;
	vector<vector<int> > qdata;

	InitTSOneStepByScan(inputFilename, queryFile, columnAv, dim, topk,
			bldLoader, Lvec, Kvec, resIdx, dist, Xtrn, Ytrn, qdata);

	vector<vector<int> > resIdxQueryOne;
	resIdxQueryOne.clear();
	resIdxQueryOne.push_back(resIdx[0]);

	LOOCVPredictor<int>* looPrd;

	//if (weighted) {
	//	looPrd = new LOOCVPredictor(Lvec, Kvec, qdata[0],Xtrn, Ytrn,  dist, 5, 1, 1);
	//} else {
	//	looPrd = new LOOCVPredictor(Lvec, Kvec, qdata[0], Xtrn, Ytrn,  5, 1, 1);
	//}

	looPrd = new LOOCVPredictor<int>(Lvec, Kvec, 5, 1, 1);

	looPrd->makeItrPrediction(qdata[0],dist,bldLoader,resIdxQueryOne,14);

	//looPrd->makeItrPrediction(qdata[0],dist,bldLoader,resIdx,14);
	delete looPrd;

}

/*

void WrapperPredictor::TSConRegPredictor(string inputFilename,
		string queryFile, int columnAv, int dim, int topk, int steps,bool weighted){
		BladeLoader<int> bldLoader;
		vector<int> Lvec;
		vector<int> Kvec;

		vector<vector<int> > resIdx;
		vector<vector<int> > dist;
		vector<vector<vector<int> > > Xtrn;
		vector<vector<int> > Ytrn;
		vector<vector<int> > qdata;

		//InitTSOneStepQueryByScan(inputFilename, queryFile, columnAv, dim, topk,
		//		bldLoader, Lvec, Kvec, resIdx, dist, Xtrn, Ytrn, qdata);

		bldLoader.loadData(inputFilename.c_str(), columnAv, &atoi);
		cout << "load data item:" << _bldLoader.data.size() << endl;
		initQuery(queryFile.c_str(), qdata);

		for(int i=0;i<qdata.size();i++){
		RtrDTWScan(bldLoader.data, qdata, dim, topk, resIdx, dist);
		}


		vector<vector<int> > resIdxQueryOne;
		resIdxQueryOne.clear();
		resIdxQueryOne.push_back(resIdx[0]);

		RegPredictor<int> *regPrd=new RegPredictor<int>(Lvec, Kvec);
		if(weighted){
		regPrd->makeItrPrediction(qdata[0],dist,bldLoader,resIdxQueryOne,14);
		}else{
			regPrd->makeItrPrediction(qdata[0],bldLoader,resIdxQueryOne,14);
		}

		delete regPrd;
}
*/

