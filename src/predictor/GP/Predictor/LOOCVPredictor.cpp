/*

  LOOCVPredictor.cpp
 *
 *  Created on: Apr 7, 2014
 *      Author: zhoujingbo


#include "LOOCVPredictor.h"
#include "../gp/optimisation/SCGModelTrainer.h"
#include "UtlPredictor.h"



 LOOCVPredictor::LOOCVPredictor()
 {
 // TODO Auto-generated constructor stub

 }

LOOCVPredictor::~LOOCVPredictor() {
	// TODO Auto-generated destructor stub
}


void LOOCVPredictor<T>::setDefaultWeightVec() {

	weightVec.clear();
	weightVec.resize(numL);
	for (int i = 0; i < numL; i++) {
		weightVec[i] = vec().ones(Kvec[numK - 1]);
	}

}

*
	 * set the weight based on DTW for each training data item

void LOOCVPredictor<T>::setWeightVec(const vector<vector<T> > weight) {
		weightVec.clear();
		weightVec.resize(numL);
		for (int i = 0; i < numL; i++) {
			vec weightSlice;
			Vector2Vec(weight[i], weightSlice);
			weightVec[i] = weightSlice;
		}
}



*
 * make prediction after configuring the data

void LOOCVPredictor<T>::inferPrediction(const vector<T> & dataXtst,
		const vector<vector<vector<T> > >& dataXtrn,
		const vector<vector<T> >& dataYtrn) {
	setXYtrn(dataXtrn, dataYtrn);
	setXtst(dataXtst);
	inferPrediction();
}


void LOOCVPredictor<T>::inferPrediction() {

	cout<<"iterativeSteps:"<<iterativeSteps<<endl;
	for (int i = 0; i < numL; i++) {

		mat& XtrnSlice = Xtrn[i];
		vec& YtrnSlice = Ytrn[i];
		vec& weightSlice = weightVec[i];
		mat XtstSlice = Xtst.cols(Xtst.n_cols - Lvec[i], Xtst.n_cols - 1); //only record the Xtst with the maximum length (maximum dimensions)
		int dim = Lvec[i];

		for (int j = 0; j < numK; j++) {
			mat XtrnSgm = XtrnSlice.rows(0, Kvec[j] - 1);
			vec YtrnSgm = YtrnSlice.rows(0, Kvec[j] - 1);
			vec weightSgm = weightSlice.rows(0, Kvec[j] - 1);

			vec meanStep = vec().zeros(1);
			vec varStep = vec().zeros(1);

			PrdMat[i][j] = inferPredictionSgm(XtrnSgm, YtrnSgm, weightSgm,
					XtstSlice, dim, sumCFMat[i][j], *(gaussianCFMat[i][j]),
					meanStep, varStep);

			assert(meanStep.n_rows==1&&varStep.n_rows==1);
			mean[i][j].insert_rows(mean[i][j].n_rows, meanStep);
			var[i][j].insert_rows(var[i][j].n_rows, varStep);

		}
	}

	iterativeSteps++;
}




 * make prediction based on one segement of the data, i.e. corresponding to one (l,k) element in predictor matrix

LOOCVGP* LOOCVPredictor<T>::inferPredictionSgm(mat& XtrnSgm, vec& YtrnSgm,
		vec & weightSgm, mat& XtstSlice, int dim, CovarianceFunction& _sumCF,
		CovarianceFunction& _gaussianCF, vec & _meanSgm, vec& _varSgm) {

	LOOCVGP* loocv = new LOOCVGP(dim, 1, XtrnSgm, YtrnSgm, _sumCF, weightSgm);
	SCGModelTrainer scggp(*loocv);
	scggp.Train(5);

	loocv->makePredictions(_meanSgm, _varSgm, XtstSlice, _gaussianCF);

	return loocv;
}


void LOOCVPredictor<T>::inferItrPrediction(BladeLoader<T> & bldLoader,
			const vector<vector<int> >& resIdx, const int steps) {
		double m, v;
		cout	<< "============start multiple steps ahead iterative prediction======================================"
				<< endl;

		inferFirstItrPrediction();
		getPredictonResult(m, v);
		cout << "0 step: mean is:" << m << " variance is:" << v << endl;

		for (int s = 1; s < steps; s++) {
			vector<vector<int> > YNextTrn;
			bldLoader.retrieveYNextTrn(resIdx, Lvec, s, YNextTrn);
			inferNextItrPrediction(YNextTrn);
			getPredictonResult(s, m, v);
			cout << "the " << s << " step:" << "mean is:" << m
					<< " variance is:" << v << endl;
		}

	}

	*
	 * note: this is first prediction,  there is no shiftConcatXtrn() and updateYtrn();

void LOOCVPredictor<T>::inferFirstItrPrediction() {
		iterativeSteps = 0;
		inferItrPrediction();
}

	*
	 * note: makeNextItrPrediction is followed by makeFirstPrediction. It is not reasonable to call makeNextItrPrediction
	 * without calling makeFirstItrPrediction

	void LOOCVPredictor<T>::inferNextItrPrediction(const vector<vector<T> >& dataYtrn) {
		shiftConcatXtrn();
		updateYtrn(dataYtrn);
		inferItrPrediction();
	}



void LOOCVPredictor<T>::inferItrPrediction(const vector<T> & dataXtst,
			BladeLoader<T> & bldLoader, const vector<vector<int> > resIdx,
			const int steps) {

		//set initial data
		vector<vector<vector<T> > > XtrnRtr;
		vector<vector<T> > YTrnRtr;
		bldLoader.retrieveXYtrn(resIdx, Lvec, XtrnRtr, YTrnRtr);
		setData(dataXtst, XtrnRtr, YTrnRtr);

		inferItrPrediction(bldLoader, resIdx, steps);

}



 * make perdiction based on the Xtrn and Ytrn, the Xtrn and Ytrn should have been configured. This also use error propagation for iterative prediction

void LOOCVPredictor<T>::inferItrPrediction() {

	for (int i = 0; i < numL; i++) {
		mat& XtrnSlice = Xtrn[i];
		vec& YtrnSlice = Ytrn[i];
		vec& weightSlice = weightVec[i];
		//mat XtstSlice = Xtst.cols(Xtst.n_cols - Lvec[i], Xtst.n_cols - 1); //only record the Xtst with the maximum length (maximum dimensions)


		int dim = Lvec[i];

		for (int j = 0; j < numK; j++) {
			mat XtrnSgm = XtrnSlice.rows(0, Kvec[j] - 1);
			vec YtrnSgm = YtrnSlice.rows(0, Kvec[j] - 1);
			vec weightSgm = weightSlice.rows(0, Kvec[j] - 1);

			vec meansgm(1);// = vec().zeros(1);
			vec varSgm(1);// = vec().zeros(1);


			if(NULL == PrdMat[i][j]||iterativeSteps == 0){
				PrdMat[i][j] = new LOOCVGP(dim, 1, XtrnSgm, YtrnSgm, sumCFMat[i][j], weightSgm);
			}

			inferItrPredictionSgm(PrdMat[i][j], XtrnSgm, YtrnSgm, XtstMeanItr[i][j], XtstVarItr[i][j],
					*(gaussianCFMat[i][j]), meansgm, varSgm);

			assert(meansgm.n_rows==1&&varSgm.n_rows==1);

			mean[i][j].insert_rows(iterativeSteps, meansgm);
			var[i][j].insert_rows(iterativeSteps, varSgm);

		}
	}

	iterativeSteps++;
}

void LOOCVPredictor<T>::inferItrPredictionSgm( MetaGP* metaGP, mat& XtrnSgm, vec& YtrnSgm,
	vec& _XtstMeanItrSgm, mat& _XtstVarItrSgm,
	CovarianceFunction& _gaussianCF, vec & _meanSgm, vec& _varSgm){

	metaGP->setXYtrn(XtrnSgm, YtrnSgm);
	SCGModelTrainer scggp(*metaGP);
	scggp.Train(5);

	metaGP->makeIterativePredictions(_XtstMeanItrSgm,_XtstVarItrSgm,_gaussianCF,_meanSgm,_varSgm);

}



*/
