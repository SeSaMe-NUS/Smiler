/*


#include "RegPredictor.h"
#include "../gp/gaussianProcesses/KnnReg.h"


 * RegPredictor.cpp
 *
 *  Created on: Apr 3, 2014
 *      Author: zhoujingbo




RegPredictor::RegPredictor(vector<int> _Lvec, vector<int> _Kvec):Predictor(_Lvec,_Kvec)
{
	// TODO Auto-generated constructor stub
	initPrdMat();

}

RegPredictor::~RegPredictor()
{
	// TODO Auto-generated destructor stub

	clearPrdMat();


}

void RegPredictor<T>::makePrediction(){


		for(int i=0;i<numL;i++){

			mat& XtrnSlice = Xtrn[i];
			vec& YtrnSlice = Ytrn[i];

			for(int j=0;j<numK;j++){
				mat XtrnSegm = XtrnSlice.rows(0,Kvec[j]-1);
				vec YtrnSegm = YtrnSlice.rows(0,Kvec[j]-1);

				//noted: knnReg only records the reference of XtrnSgm and YtrnSgm
				KnnReg* knnreg = new KnnReg(XtrnSegm,YtrnSegm);
				// Compute the posterior GP and store the result in mean and var
				vec meanStep=vec().zeros(1);
				vec varStep = vec().zeros(1);
				knnreg->makePredictions(meanStep, varStep);

				assert(meanStep.n_rows==1&&varStep.n_rows==1);
				mean[i][j].insert_rows(mean[i][j].n_rows,meanStep);
				var[i][j].insert_rows(var[i][j].n_rows,meanStep);

				PrdMat[i][j] = knnreg;
			}
		}
		iterativeSteps++;
}


void RegPredictor<T>::clearPrdMat(){
	for (int i = 0; i < numL; i++) {
			for (int j = 0; j < numK; j++) {
				if (NULL != PrdMat[i][j]) {
					delete PrdMat[i][j];
				}
				PrdMat[i][j] = NULL;

			}
			//PrdMat[i].clear();
		}
		//PrdMat.clear();
}

void RegPredictor<T>::initPrdMat(){

	PrdMat.resize(numL);
	for(int i=0;i<numL;i++){
		PrdMat[i].resize(numK);
		for(int j=0;j<numK;j++){
			PrdMat[i][j] = NULL;
		}
	}

}

void RegPredictor<T>::updateXtst(){
	Xtst.shed_col(0);
	vec tstVec(Xtst.n_rows);

	double meanPre, varPre;
	getPredictonResult(meanPre,varPre);
	tstVec.fill(meanPre);
	Xtst.insert_cols(Xtst.n_cols,tstVec);
}

void RegPredictor<T>::makeItrPrediction(const vector<vector<T> >& dataYtrn) {
		shiftConcatXtrn();
		updateYtrn(dataYtrn);
		updateXtst();
		clearPrdMat();
		makePrediction();
}



*/
