/*
 * WrapperSemiLazyPredictor.h
 *
 *  Created on: Apr 4, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERSEMILAZYPREDICTOR_H_
#define WRAPPERSEMILAZYPREDICTOR_H_

#include <string>
#include <vector>
#include "../../tools/BladeLoader.h"
#include "../../searcher/WrapperScan.h"
using namespace std;

class WrapperPredictor {
public:
	WrapperPredictor();
	virtual ~WrapperPredictor();

	void TSOneStepRegPredictor(string inputFilename, string queryFile,int columnAv, int dim, int k,bool weighted);
	void TSOneStepLOOCVPredictor(string inputFilename, string queryFile,int columnAv, int dim, int topk, bool weighted);
	void TSMulStepsRegPredictor(string inputFilename, string queryFile,int columnAv, int dim, int topk, int steps,bool weighted);
	void TSMulStepsLOOCVPredictor(string inputFilename,	string queryFile, int columnAv, int dim, int topk, int steps, bool weighted);

private:

	void initQuery(const char* queryFile,vector<vector<int> > & _qdata);
	void initLK(const int L, const int K, vector<int>& _Lvec, vector<int>& _Kvec);

	template <class T>
	void RtrDTWScan(vector<T>& data, vector<vector<T> >& qdata, int dim, int topk,
			vector<vector<int> >& _resIdx, vector<vector<float> >& _dist){

			long time = 0;
			long start = clock();
			WrapperScan wrapperDTW;
			//wrapperDTW.DTWQuery(data, qdata, dim, topk, _resIdx,_dist);
			long end = clock();
			time = end - start;
			cout << "the running time of GPU scan is:" << (double) time/CLOCKS_PER_SEC << endl;

	}

	void InitTSOneStepByScan(string inputFilename, string queryFile, int columnAv, int dim, int topk,
			BladeLoader<int>& _bldLoader, vector<int>& _Lvec,
			vector<int>& _Kvec, vector<vector<int> >& _resIdx, vector<vector<float> >& _dist,vector<vector<vector<int> > >& _Xtrn,
			vector<vector<int> >& _Ytrn, vector<vector<int> >& _qdata);

	void InitTSOneStepByScan(string inputFilename,
			string queryFile, int columnAv, int dim, int topk, vector<int>& _Lvec,
			vector<int>& _Kvec, vector<vector<float> >& _dist, vector<vector<vector<int> > >& _Xtrn,
			vector<vector<int> >& _Ytrn, vector<vector<int> >& _qdata) ;



};

#endif /* WRAPPERSEMILAZYPREDICTOR_H_ */
