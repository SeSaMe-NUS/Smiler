/*
 * Predictor.h
 *
 *  Created on: Apr 2, 2014
 *      Author: zhoujingbo
 */

#ifndef PREDICTOR_H_
#define PREDICTOR_H_

#include "UtlPredictor.h"
#include <assert.h>
#include <cmath>
#include <time.h>
#include <stdlib.h>
#include "../../../tools/BladeLoader.h"

//for custom
#include "../gp/gaussianProcesses/MetaGP.h"
#include "UtlPredictor.h"

//this is specialized for one sensor.
template<class T>
class Predictor {
public:
	//Predictor(vector<int> _Lvec, vector<int> _Kvec);
	Predictor(vector<int>& Lvec, vector<int>& Kvec,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn, const vector<T> & dataXtst) {

		this->Lvec = Lvec;
		this->Kvec = Kvec;

		numL = Lvec.size();
		numK = Kvec.size();
		initialParameter();

		setXYtrn(dataXtrn, dataYtrn);

		setXtst(dataXtst);



	}

	//constructor for multiple queries, only create the predictor with the common configuration
	Predictor(vector<int>& Lvec, vector<int>& Kvec) {
		this->Lvec = Lvec;
		this->Kvec = Kvec;

		numL = Lvec.size();
		numK = Kvec.size();

		initialParameter();
	}

	virtual ~Predictor() {
	}

public:
	/**
	 * with the data configuration by constructor function
	 */
	void makePrediction() {

		inferPrediction();

	}

	void makePrediction(const vector<T> & dataXtst,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn) {

		setDefaultWeightVec();
		inferPrediction(dataXtst, dataXtrn, dataYtrn);

	}

	/**
	 * dataXtst: Xtst with the maximum length
	 */
	void makePrediction(const vector<T> & dataXtst,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn,	const vector<vector<float> >& dist) {

		setWeightVec(dist);
		inferPrediction(dataXtst, dataXtrn, dataYtrn);

	}


	/**
	 * TODO:
	 * 	   to make prediction based on 1-step head prediction
	 */
	void selfCorrWeightPrdMat(T Ytst){
		selfCorrWeightPrdMat(Ytst, 0);
	}


	vector<vector<double> >& getWeightPrdMat(){
		return this->weightPrdMat;
	}

	vector<vector<int> >& getWeightPrdMat_sleepRecord(){
		return this->weightPrdMat_sleepRecord;
	}
	vector<vector<int> >& getWeightPrdMat_prdRecord(){
		return this->weightPrdMat_prdRecord;
	}
	vector<vector<int> >& getWeightPrdMat_recoverStep(){
		return this->weightPrdMat_recoverStep;
	}

	void selfCorrWeightPrdMat(double Ytst, vector<vector<double> >& mean, vector<vector<double> >& var){

		vector<vector<double> > wpm(numL);
		double wpm_sum=0;
		for (int i = 0; i < numL; i++) {
			wpm[i].resize(numK);
			for (int j = 0; j < numK; j++) {

				if(this->weightPrdMat[i][j]>=weightPrdMat_cutoff){

					wpm[i][j] = computePrdLikelihood(Ytst, mean[i][j], var[i][j]);
					wpm_sum+=wpm[i][j];

				}else{ 	wpm[i][j] = 0; }
			}

		}


		if (wpm_sum<=std::numeric_limits< double >::min()){
			wpm_sum = (std::numeric_limits< double >::min());//to avoid divide by zero

			//all of them is zero, select one predictor and let others sleep
			for(int i=0;i<numL&&wpm_sum<1;i++){
				for(int j=0;j<numK&&wpm_sum<1;j++){
					if(this->weightPrdMat[i][j]>=weightPrdMat_cutoff){
						wpm[i][j]=1;
						wpm_sum += 1;
						break;
					}
				}
			}
		}




		//update the weightPrdMat
		double weightPrdMat_sum = 0;
		for(int i=0;i<numL;i++){
			for (int j=0;j<numK;j++){
				this->weightPrdMat[i][j]=this->weightPrdMat[i][j]*(1-weightPrdMat_learnRate)+(wpm[i][j]/wpm_sum)*weightPrdMat_learnRate;
				weightPrdMat_sum+=this->weightPrdMat[i][j];
			}
		}

		double weightSum_new = weightPrdMat_sum;
		double weightCutoff=weightPrdMat_cutoff*weightPrdMat_sum;
		double label_recover = -1;//the label for weightPrdMat to recover this expert
		int count_recover = 0;


		for(int i=0;i<numL;i++){
			for(int j=0;j<numK;j++){
				//make this expert sleep if its weight smaller than cutoff value
				if(this->weightPrdMat[i][j]<weightCutoff){
					weightSum_new-=weightPrdMat[i][j];
					weightPrdMat[i][j] = 0;

					if(weightPrdMat_sleepRecord[i][j]==0){// befor going to sleep, make some preparation for waking up

						if(weightPrdMat_prdRecord[i][j]==0){//after recovery, if no prediction successfully makes, double the weightPrdMat_recoverStep

							//remove noise
							int wpm_rs = weightPrdMat_recoverStep[i][j] - (weightPrdMat_recoverStep[i][j]%(UtlPredictor_namespace::weightPrdMat_minRecoverStep));
							//increase the recoverStep
							wpm_rs = wpm_rs*(UtlPredictor_namespace::weightPrdMat_baseRecoverStep);

							wpm_rs = (wpm_rs<=UtlPredictor_namespace::weightPrdMat_maxRecoverStep)?
															wpm_rs: (UtlPredictor_namespace::weightPrdMat_minRecoverStep);

							//add noise
							wpm_rs =wpm_rs + (rand() % (UtlPredictor_namespace::weightPrdMat_minRecoverStep));

							weightPrdMat_recoverStep[i][j]=wpm_rs;

						}else{

							//remove noise
							int wpm_rs = weightPrdMat_recoverStep[i][j] - (weightPrdMat_recoverStep[i][j]%(UtlPredictor_namespace::weightPrdMat_minRecoverStep));

							//after recovery, if prediction successfully makes, decrease the weightPrdMat_recoverStep
							wpm_rs = wpm_rs/((int)powf(UtlPredictor_namespace::weightPrdMat_baseRecoverStep,weightPrdMat_prdRecord[i][j]));
							wpm_rs = (wpm_rs>=UtlPredictor_namespace::weightPrdMat_minRecoverStep)?
																						wpm_rs:(UtlPredictor_namespace::weightPrdMat_minRecoverStep);
							//add noise
							wpm_rs = wpm_rs+(rand() % (UtlPredictor_namespace::weightPrdMat_minRecoverStep));
							weightPrdMat_recoverStep[i][j] = wpm_rs;
						}
					}
					weightPrdMat_sleepRecord[i][j]++;//going to sleep, record this step

					//check whether to recover this expert if weightPrdMat_sleepRecord[i][j]>=weightPrdMat_recoverStep[i][j]
					if(weightPrdMat_sleepRecord[i][j]>weightPrdMat_recoverStep[i][j]){
						//weightPrdMat[i][j] = this->weightPrdMat_cutoff*1.01;
						//weightSum_new+=this->weightPrdMat_cutoff*1.01;
						count_recover++;
						weightPrdMat[i][j] = label_recover;//label this expert to be recovered

						weightPrdMat_sleepRecord[i][j] = 0;//prepare for next sleep
						weightPrdMat_prdRecord[i][j]=0;//prepare to record the predictio step
					}
				}else{
					weightPrdMat_prdRecord[i][j]++;//increase prediction step after successfully avoid the pitfall of sleep
				}
			}
		}


		//x/(S+Ax)=c=> x=cS/(1-Ac), c is cutoff, A is the number of experts to be recovered
		double weight_recover = weightPrdMat_cutoff*weightSum_new/(1-count_recover*weightPrdMat_cutoff);
		weightSum_new = weightSum_new+count_recover*weight_recover;

		if(weightSum_new<= (std::numeric_limits< double >::min())){
			weightSum_new = (std::numeric_limits< double >::min());//to avoid divide by zero
		}


		//re-normalize the weightPrdMat
		for(int i=0;i<numL;i++){
			for(int j=0;j<numK;j++){
				if(this->weightPrdMat[i][j]==label_recover){
					this->weightPrdMat[i][j] = weight_recover;
				}
				this->weightPrdMat[i][j]/=weightSum_new;
			}
		}
	}


	/**
	 * TODO:
	 *     to make prediction based on multiple step-ahead prediction
	 *
	 *     adaptive ajust the weight for each prediciton expert
	 *     1. if the weight is smaller than weightPrdMat_cutoff, make this expert sleep
	 *     2. after sleep weightPrdMat_recoverStep, recover this expert
	 *     3. if this expert fall into sleep again, set the recover step for this expert
	 *     	  3.1. if this expert falls into weightPrdMat_cutoff without any prediction step, increase the recoveryStep exponentially
	 *     	  3.2. if the expert makes n steps, decrease the recover step exponentially
	 *     	  3.3. to recover the expert, give it weight with the equation
	 *     	       x/(S+Ax)=c=> x=cS/(1-Ac), c is cutoff ratio, A is the number of experts to be recovered
	 *@param: Ytst:the true result
	 *@param: step: to do self-correction based on the which step-head of prediction
	 * int step
	 *
	 *
	 */
	//be here!!!
	void selfCorrWeightPrdMat(T Ytst, int step){
		vector<vector<double> > meanMat, varMat,weightMat;
		getPredictionMatrix( step, meanMat,  varMat, weightMat);
		selfCorrWeightPrdMat(Ytst,meanMat,varMat);

	}


	/**
	 * in this function, we compute the the likelihood of each prediction for one prediction expert
	 */
	double computePrdLikelihood(T Ytst, T prdMean, T prdVar){

		return 1/sqrt(2*M_PI*prdVar)*exp(-((Ytst-prdMean)*(Ytst-prdMean)/(2*prdVar)));
	}



	void makeItrPrediction(const vector<T> & dataXtst,
			BladeLoader<T> & bldLoader, const vector<vector<int> >& resIdx,
			const int steps) {

		setDefaultWeightVec();
		inferItrPrediction(dataXtst, bldLoader, resIdx, steps);

	}

	void makeItrPrediction(const vector<T> & dataXtst,
			const vector<vector<float> >& weight, BladeLoader<T> & bldLoader,
			const vector<vector<int> >& resIdx, const int steps) {

		setWeightVec(weight);
		inferItrPrediction(dataXtst, bldLoader, resIdx, steps);

	}

	/**
	 * _meanRes:
	 * _varRes:
	 * step:step-ahead prediction result
	 */
	void getPredictonResult(int step, double & _meanRes, double & _varRes) {
		double C = 0;
		double meanSum = 0;
		double varSum = 0;
		for (int i = 0; i < numL; i++) {

			for (int j = 0; j < numK; j++) {
				double cij = weightPrdMat[i][j];

				C += cij;
				//the prediction result of each expert is stored in mean and var
				assert(step < mean[i][j].n_rows && step < var[i][j].n_rows);

				meanSum = cij * (mean[i][j](step)) + meanSum;
				varSum = cij * (var[i][j](step)) + varSum;



			}
		}

		_meanRes = meanSum / C;
		_varRes = varSum / C;
	}

	void getPredictionMatrix(int step, vector<vector<double> >& meanMat, vector<vector<double> >& varMat,vector<vector<double> >& weightMat){

		meanMat.resize(numL);
		varMat.resize(numL);
		weightMat.resize(numL);

		for(int i=0;i<numL;i++){
			meanMat[i].resize(numK);
			varMat[i].resize(numK);
			weightMat[i].resize(numK);

			for(int j=0;j<numK;j++){
				meanMat[i][j]=mean[i][j](step);
				varMat[i][j] = var[i][j](step);
				weightMat[i][j]=weightPrdMat[i][j];
			}
		}
	}

	void getPredictionMatrix(vector<vector<double> >& meanMat, vector<vector<double> >& varMat,vector<vector<double> >& weightMat){

		getPredictionMatrix(0, meanMat, varMat,weightMat);

	}

	void getPredictionResult(vec& _mean_vec, vec& _var_vec){

		double C = 0;

		_mean_vec.set_size(this->iterativeSteps);
		_mean_vec.zeros();

		_var_vec.set_size(this->iterativeSteps);
		_var_vec.zeros();

		for(int i=0;i<numL;i++){
			for(int j=0;j<numK;j++){

				double cij = weightPrdMat[i][j];
				C+=cij;

				assert(this->iterativeSteps == mean[i][j].n_rows && this->iterativeSteps == var[i][j].n_rows);

				_mean_vec = cij*mean[i][j]+_mean_vec;
				_var_vec = cij*var[i][j]+_var_vec;
			}
		}

		_mean_vec = _mean_vec/C;
		_var_vec = _var_vec / C;

	}

	//default prediction result, i.e. step = 0;
	void getPredictonResult(double & _meanRes, double & _varRes) {
		getPredictonResult(0, _meanRes, _varRes);
	}

	void setData(const vector<T> & dataXtst,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn) {

		setXYtrn(dataXtrn, dataYtrn);
		setXtst(dataXtst);
	}

	void setDisplay(bool display){
		this->display = display;
	}

protected:
	//================================= abstract interface of Prediction part ===========================
	virtual void inferPrediction()=0;
	virtual void inferItrPrediction() = 0;


	//for direct prediction, nothing to do with Xtrn
	//for iterative prediction, call "shiftConcatXtrn()"
	//implement for different prediction functions
	virtual void updateNextXtrn()=0;
	//


protected:
//=================================Prediction part ===========================


	/**
	 * make prediction after configuring the data
	 */
	void inferPrediction(const vector<T> & dataXtst,
			const vector<vector<vector<T> > >& dataXtrn,
			const vector<vector<T> >& dataYtrn) {

		resetResultContainer();
		setData(dataXtst,dataXtrn,dataYtrn);
		inferPrediction();

	}


	/**
	 * Xtst with multiple dimensions and topK
	 * this function makes multiple steps ahead prediction with inputing configuration data
	 */
	void inferItrPrediction(const vector<T> & dataXtst,
			BladeLoader<T> & bldLoader, const vector<vector<int> >& resIdx,
			const int steps) {

		//set initial data
		vector<vector<vector<T> > > XtrnRtr;
		vector<vector<T> > YTrnRtr;
		resetResultContainer();

		bldLoader.retrieveXYtrn(resIdx, this->Lvec, XtrnRtr, YTrnRtr);
		setData(dataXtst, XtrnRtr, YTrnRtr);

		inferItrPrediction(bldLoader, resIdx, steps);

	}

	void inferItrPrediction(BladeLoader<T> & bldLoader,
			const vector<vector<int> >& resIdx, const int steps) {
		double m, v;
		if(display){
		cout
				<< "============start multiple steps ahead iterative prediction======================================"
				<< endl;
		}


		inferFirstItrPrediction();
		this->getPredictonResult(m, v);

		if(display){
		cout << "0 step: mean is:" << m << " variance is:" << v << endl;
		}

		for (int s = 1; s < steps; s++) {
			vector<vector<T> > YNextTrn;
			bldLoader.retrieveYNextTrn(resIdx, this->Lvec, s, YNextTrn);
			inferNextItrPrediction(YNextTrn);
			this->getPredictonResult(s, m, v);


			if(display){
			cout << "the " << s << " step:" << "mean is:" << m
					<< " variance is:" << v << endl;
			}
		}

	}

	/**
	 * note: this is first prediction,  there is no shiftConcatXtrn() and updateYtrn();
	 */
	void inferFirstItrPrediction() {
		this->iterativeSteps = 0;
		inferItrPrediction();
	}


	/**
	 * note: makeNextItrPrediction is followed by makeFirstPrediction. It is not reasonable to call makeNextItrPrediction
	 * without calling makeFirstItrPrediction
	 */
	void inferNextItrPrediction(const vector<vector<T> >& dataYtrn) {

		//for direct prediction, nothing to do with Xtrn
		//for iterative prediction, call "shiftConcatXtrn()"
		//implement for different prediction functions
		updateNextXtrn();

		updateYtrn(dataYtrn);

		inferItrPrediction();
	}



//============================data operation part=============================================
	void setXtrn(const vector<vector<vector<T> > >& dataXtrn) {
		int lenXtrn = dataXtrn.size();
		assert(lenXtrn == numL);
		//check, number of slices of training data should be equal to number of different dimensions
		Xtrn.clear();
		Xtrn.resize(lenXtrn);

		for (int i = 0; i < lenXtrn; i++) {
			mat slice;
			TwoDVector2Mat(dataXtrn[i], slice);
			Xtrn[i] = slice;
		}
	}

	void setYtrn(const vector<vector<T> >& dataYtrn) {

		int lenYtrn = dataYtrn.size();
		assert(lenYtrn == numL);
		//check, number of slices of training data should be equal to number of different dimensions
		Ytrn.clear();
		Ytrn.resize(lenYtrn);

		for (int i = 0; i < lenYtrn; i++) {
			vec sliceYtrn;
			Vector2Vec(dataYtrn[i], sliceYtrn);
			Ytrn[i] = sliceYtrn;
		}
	}

	void setXYtrn(const vector<vector<vector<T> > > & dataXtrn,
			const vector<vector<T> >& dataYtrn) {

		setXtrn(dataXtrn);
		setYtrn(dataYtrn);

	}

	void setXtst(const vector<T>& dataXtst) {

		int XtstLen = dataXtst.size();
		//the length of the queries should be equal the maximum dimensions for training data
		assert(XtstLen == Lvec[Lvec.size() - 1]);
		Xtst.set_size(1, dataXtst.size());

		for (int i = 0; i < dataXtst.size(); i++) {
			Xtst(0, i) = dataXtst[i];
		}

		//set data for iterative prediction
		for (int i = 0; i < numL; i++) {
			int d = Lvec[i];

			for (int j = 0; j < numK; j++) {

				XtstMeanItr[i][j] = vec().zeros(d);
				XtstVarItr[i][j] = mat().zeros(d, d);

				for (int l = 0; l < d; l++) {
					XtstMeanItr[i][j](l) = Xtst(0, l);
				}
			}
		}

	}

	/**
	 * shift and concatenate: first delete the first column of Xtrn, and then add the Ytrn as the last column of Xtrn
	 */
	void shiftConcatXtrn() {

		for (int i = 0; i < numL; i++) {
			mat& XtrnSlice = Xtrn[i];
			vec YtrnSlice = Ytrn[i]; //NOTE: this is not reference since the Ytrn is inserted into the mat, and Ytrn will be updated later
			XtrnSlice.shed_col(0);
			XtrnSlice.insert_cols(XtrnSlice.n_cols, YtrnSlice);
		}
	}

	/**
	 * insert new Ytrn into the array
	 */
	void updateYtrn(const vector<vector<T> >& dataYtrn) {
		setYtrn(dataYtrn);
	}

	void print2DVector(const vector<vector<T> >& Vec2D){

		cerr<<" 2D std vector with row:"<<Vec2D.size()<<endl;
		for(int i=0;i<Vec2D.size();i++){
			for(int j=0;j<Vec2D[i].size();j++){
				cerr<<" "<<Vec2D[i][j];
			}
			cerr<<endl;
		}

	}

	void printweightPrdMat(){
		cout << "print weightPrdMat" << endl;
		for (int i = 0; i < this->numL; i++) {

			for (int j = 0; j < this->numK; j++) {
				double cij = this->weightPrdMat[i][j];

				printf("[%d][%d]:%f ", i, j, cij);

			}
			printf("\n");
		}
	}

protected:
	/**
	 * _dataMat: the output result
	 */
	void TwoDVector2Mat(const vector<vector<T> > & dataVecs, mat& _dataMat) {
		int row = dataVecs.size();
		int col = dataVecs[0].size();

		_dataMat.clear();
		_dataMat.resize(row, col);
		//mat res(row,col);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				_dataMat(i, j) = dataVecs[i][j];
			}
		}

	}

	void Vector2Vec(const vector<T>& dataVector, vec& _dataVec) {
		int col = dataVector.size();

		_dataVec.clear();
		_dataVec.resize(col);
		for (int i = 0; i < col; i++) {
			_dataVec(i) = dataVector[i];
		}
	}

	void weight_Vector2Vec(const vector<float>& dataVector, vec& _dataVec) {
		int col = dataVector.size();

		_dataVec.clear();
		_dataVec.resize(col);
		for (int i = 0; i < col; i++) {
			_dataVec(i) = dataVector[i];
		}
	}

	//===========================parameter configuration part=======================
	void initialParameter() {

		initWeightParameters();
		initResultContainer();

		display = false;
		weightPrdMat_learnRate= UtlPredictor_namespace::weightPrdMat_learnRate;
		weightPrdMat_cutoffRate = UtlPredictor_namespace::weightPrdMat_cutoffRate;
		weightPrdMat_cutoff = (1./(numL*numK))*weightPrdMat_cutoffRate;
	}

	void initWeightParameters(){

		weightPrdMat.clear();
		weightPrdMat.resize(numL);

		weightPrdMat_sleepRecord.clear();
		weightPrdMat_sleepRecord.resize(numL);
		weightPrdMat_prdRecord.clear();
		weightPrdMat_prdRecord.resize(numL);
		weightPrdMat_recoverStep.clear();
		weightPrdMat_recoverStep.resize(numL);

		setDefaultWeightVec();

		for (int i = 0; i < numL; i++) {
			weightPrdMat[i].resize(numK);

			weightPrdMat_sleepRecord[i].resize(numK);
			weightPrdMat_prdRecord[i].resize(numK);
			weightPrdMat_recoverStep[i].resize(numK);

			for(int j=0;j<numK;j++){
				weightPrdMat[i][j] = 1./(numK*numL);//normalized
				weightPrdMat_sleepRecord[i][j] = 0;
				weightPrdMat_prdRecord[i][j]=0;
				weightPrdMat_recoverStep[i][j]=UtlPredictor_namespace::weightPrdMat_minRecoverStep;
			}
		}
	}

	void initResultContainer() {
		mean.clear();
		mean.resize(numL);

		var.clear();
		var.resize(numL);

		XtstMeanItr.clear();
		XtstMeanItr.resize(numL);
		XtstVarItr.clear();
		XtstVarItr.resize(numL);

		iterativeSteps = 0;

		for (int i = 0; i < numL; i++) {
			mean[i].resize(numK);
			var[i].resize(numK);
			XtstMeanItr[i].resize(numK);
			XtstVarItr[i].resize(numK);

			//initialize the item in the matrix
			for (int j = 0; j < numK; j++) {
				mean[i][j] = vec().zeros(0);
				var[i][j] = vec().zeros(0);
			}
		}

	}

	void resetResultContainer() {
		initResultContainer();
	}

	// the default value is 1
	void setDefaultWeightVec() {

		weightVec.clear();
		weightVec.resize(this->numL);
		for (int i = 0; i < this->numL; i++) {
			weightVec[i] = vec().ones(this->Kvec[this->numK - 1]);
		}

	}

	/**
	 * set the weight based on DTW for each training data item
	 *
	 * for each weight, we have w_i = 1/(dtw(x_0,x_i)+1)
	 */
	void setWeightVec(const vector<vector<float> >& dist) {
		weightVec.clear();
		weightVec.resize(this->numL);
		for (int i = 0; i < this->numL; i++) {
			vec weightSlice;
			weight_Vector2Vec(dist[i], weightSlice);
			weightVec[i] = 1/(sqrt(weightSlice)+1);
		}
	}

public:
	int numL; //the number of training data with different L
	int numK; //the number of training data with differnt K
	//select the tailed (i.e. latest) L elements as test data, with Lvec.size() test data
	vector<int> Lvec; //record dimensions of raining data with different L
	vector<int> Kvec; //record training data with different K

	vector<mat> Xtrn; //array of training data set, with different L and the maximum K
	vector<vec> Ytrn; //vec of label, with the maximum K

	//for mean and variance, the column is the step ahead predictions, 0 is first step, 2 is second step, ...
	vector<vector<vec> > mean; //prediction result, two dimension matrix with different k and l
	vector<vector<vec> > var; //prediction result, two dimension matrix with different k and l
	int iterativeSteps;

	vector<vector<double> > weightPrdMat; //weight of each element in PrdMat
	double weightPrdMat_learnRate;//the rate to update the this weightPrdMat with previous weightPrdMat
	double weightPrdMat_cutoffRate;//the rate to filter the expert(with dim and topk), if weightPrdMat<weightPrdMat_cutoffRate*avg_ weightPrdMat, filter this expert
	double weightPrdMat_cutoff;//1.0/numL*numK*weightPrdMat_cutoffRate;
	vector<vector<int> > weightPrdMat_prdRecord;//record how many steps this expert has made prediction after previous recovery.
	vector<vector<int> > weightPrdMat_sleepRecord;//record how many steps this expert has sleep. i.e. record how many prediction has been made after for this expert we have weightPrdMat[i][j]<weightPrdMat_cutoffRate*avg_ weightPrdMat
	vector<vector<int> > weightPrdMat_recoverStep;//after weightPrdMat_recoverStep, we reset the weight for this prediction expert as weightPrdMat_cutoff to re-join this expert to work
	vector<vec> weightVec; //weight of each Xtrn data, only with differnt L dimensions and the maximum K
	mat Xtst; //only record the Xtst with the maximum length (maximum dimensions)
	//XtstMeanItr and XtstVarItr are intermedia input for iterative multiple step ahead prediction, i.e. the
	//previous predictive distribution with the Xtst to form new input XtstMeanItr and XtstVarItr for the input
	//of the next step
	vector<vector<vec> > XtstMeanItr;
	vector<vector<mat> > XtstVarItr;

	bool display;

//	vec Ytst; //record on label value

};

#endif /* PREDICTOR_H_ */
