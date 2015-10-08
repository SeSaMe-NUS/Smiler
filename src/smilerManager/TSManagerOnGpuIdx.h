/*
 * TSManager.h
 *
 *  Created on: Jun 24, 2014
 *      Author: zhoujingbo
 */

#ifndef TSMANAGER_H_
#define TSMANAGER_H_

#include "../predictor/GP/Predictor/Predictor.h"
#include "../predictor/GP/Predictor/LOOCVPredictor.h"
#include "../predictor/GP/Predictor/MMLGPPredictor.h"
#include "../predictor/GP/Predictor/RegPredictor.h"

#include "../searcher/DataEngine/DataEngine.h"
#include "../searcher/DataEngine/DataEngineSmiler.h"

#include <cmath>



//for arma
#include "armadillo"
using namespace arma;

/**
 * in this class, for kNN search, use GPU scan
 *
 */


class TSManagerOnGpuIdx {
public:
	TSManagerOnGpuIdx() {

		//predictor = NULL;
		dataEngine = new DataEngineSmiler();
		display=false;

		//hypRange =100,  hypSill=500,  hypNugget = 320;
		hypRange =200,  hypSill=200,  hypNugget = 160;//here!!

		winDim=0;
		sc_band = 0;

	}
	virtual ~TSManagerOnGpuIdx() {

		delete dataEngine;

	}

	//squareError = square(Ytst - mean);
	//standSquareError = squareError / var;
	//negLogLikelihood = 0.5 * (standSquareError + log(var) + log(2 * datum::pi));

	double absoluteError(double Ytst, double predMean){
		return std::abs(Ytst-predMean);
	}

	vec stdVec2ArmaVec(vector<float>& std_vec){

		vec arma_vec = vec().zeros(std_vec.size());
			for(int i=0;i<std_vec.size();i++){
				arma_vec(i)=std_vec[i];
		}
		return arma_vec;
	}

	void absoluteError(vector<float>& Ytst_vec, vec& mean_vec, vec& ae){

		vec tst_vec = stdVec2ArmaVec(Ytst_vec);

		ae = abs(tst_vec-mean_vec);

	}

	double squareError(double Ytst, double predMean){
		return (Ytst-predMean)*(Ytst-predMean);
	}

	void squareError(vector<float>& Ytst_vec, vec& mean_vec, vec& se){
		vec tst_vec = stdVec2ArmaVec(Ytst_vec);
		se = square(tst_vec-mean_vec);
	}

	double standSquareError(double Ytst, double predMean, double predVar){
		double se=squareError(Ytst,predMean);
		return se*se/(predVar);
	}

	void standSquareError(vector<float>& Ytst_vec, vec& mean_vec, vec& var_vec, vec& sse){
		vec tst_vec = stdVec2ArmaVec(Ytst_vec);
		vec se = vec().zeros();
		squareError(Ytst_vec, mean_vec,se);
		sse = se/var_vec;
	}

	double negLogLikelihood(double Ytst,double predMean,double predVar){

		double sse = standSquareError(Ytst,predMean,predVar);
		return 0.5 * (sse + log(predVar) + log(2 * datum::pi));
	}

	void negLogLikelihood(vector<float> Ytst_vec, vec & mean_vec, vec& var_vec, vec& nll){
		vec tst_vec = stdVec2ArmaVec(Ytst_vec);
		vec sse = vec().zeros();
		standSquareError( Ytst_vec, mean_vec, var_vec, sse);
		vec c_vec = vec().zeros(Ytst_vec.size());
		c_vec.fill(log(2 * datum::pi));

		nll = 0.5*(sse+log(var_vec)+c_vec);
	}

	void setDisplay(bool display){
		this->display = display;
	}

	//maxOffset: do not load maxOffset data into (or say left them out of) GPU to avoid exceeding the bounding when query the reference time series//for improve
	void conf_DataEngine(vector<vector<float> >& in_bladeData_vec, int sc_band, int winDim,int maxOffset) {//for improve
				dataEngine->conf_bladeData(in_bladeData_vec,maxOffset);
				this->sc_band=sc_band;
				this->winDim = winDim;
		}

	void conf_Predictor(vector<int>& Lvec, vector<int>& Kvec) {
		this->Lvec = Lvec;
		this->Kvec = Kvec;
	}





	void printVector(vec& data){
		for(uint i=0;i<data.n_rows;i++){
			cout<<" ["<<i<<"]:"<<data(i);
		}
		cout<<endl;
	}



	int getMinRowLen(vector<vector<float> >& vec2d){

		int d=INT_MAX;

		for(int i=0;i<vec2d.size();i++){
			if(d>vec2d[i].size()){
				d=vec2d[i].size();
			}
		}

		return d;
	}


	/**
		 * TODO:
		 * use the first Lvec[max] as the test data to find topk, the query with larger than Lvec[max] and smaller than Lvec[max]+mulSteps are test Y label data
		 */
		void alignTestData_oneStep(vector<vector<float> >& query_vec,
				vector<vector<float> > & _Xtst_vec, vector<float>& _Ytst_vec,
				int xtst_len, int xtst_offset, int y_offset){
			_Xtst_vec.clear();
			_Xtst_vec.resize(query_vec.size());
			_Ytst_vec.clear();
			_Ytst_vec.resize(query_vec.size());

			for(int i=0;i<query_vec.size();i++){

				assert(query_vec[i].size()>=xtst_len+y_offset+xtst_offset);
				_Xtst_vec[i].resize(xtst_len);
				std::copy(query_vec[i].begin()+xtst_offset,query_vec[i].begin()+xtst_offset+xtst_len,_Xtst_vec[i].begin());
				_Ytst_vec[i] = query_vec[i][xtst_offset+xtst_len+y_offset];

			}
		}






	/**
	 * TODO:
	 *     make continuous prediction for each vector in Xtst_vec with GPU kNN search
	 *@param:
	 *@param: Xtst_vec -- a set of query vector for prediction
	 *@param: groupQuery_blade_map-- map the query to corresponding data blade
	 *@param: y_offset -- the h-step ahead prediction for time series prediction.
	 *                      for one step ahead prediction,y_offset==0
	 *                      for multiple steps ahead prediction, y_offset = mul_step - 1
	 *@param: weightedTrn -- whether to do weighted training process
	 *@param: selfCorr -- whether to do self-correcting prediction to determine the weight of K and L
	 *
	 *
	 */
	void TSPred_continuous_onGPUIdx(vector<vector<float> >& groupQuery_vec, int contPrdStep, vector<int>& groupQuery_blade_map,
			int y_offset, int pred_selector, bool weightedTrn, bool selfCorr){


		struct timeval tim;

		//engine for kNN search
		dataEngine->setup_contGroupQuery(
					groupQuery_vec,
					groupQuery_blade_map,
					Lvec,
					this->sc_band,this->winDim);


		vector<Predictor<float>*> prd_vec(groupQuery_vec.size());

		vector<vector<PredictionResultSet> > prdRes_set;//structure for self correction
		prdRes_set.resize(groupQuery_vec.size());


		//====
		for (int j = 0; j < groupQuery_vec.size(); j++) {

			switch (pred_selector) {
			case 0:
				prd_vec[j] = new LOOCVPredictor<float>(Lvec, Kvec, hypRange,
						hypSill, hypNugget);
				break;
			case 1:
				prd_vec[j] = new RegPredictor<float>(Lvec, Kvec);
				break;
			default:
				prd_vec[j] = new RegPredictor<float>(Lvec, Kvec);
				break;
			}

			prd_vec[j]->setDisplay(this->display);
			prdRes_set[j].resize(y_offset+1);

		}


		//===========================

		int queryItemLen = Lvec.back();
		vector<vector<float> > Xtst_vec;
		vector<float> Ytst_vec;
		//query result
		vector<XYtrnPerGroup<float> > XYtrn_vec; //query result


		int minQueryLen =  getMinRowLen(groupQuery_vec);
		int duration = minQueryLen-queryItemLen-y_offset;

		if(duration>contPrdStep&&contPrdStep!=-1){
			duration = contPrdStep;
		}
		//in total, make duration times of prediction

		bool predErr = true;
		double mae=0;//mean absolute error
		double mse=0;//mean square error
		double msse=0;//mean standard square error
		double mnll=0;//mean negative log likelihood


		double t_search_total = 0;
		double t_prd_total = 0;

		 //printf("sensorId,	step,	true value,	pred_mean,	pred_var, true_value-pred_mean\n");



		for (int ci = 0; ci < duration; ci++) {


			double mae_ci=0;//mean absolute error
			double mse_ci=0;//mean square error
			double msse_ci=0;//mean standard square error
			double mnll_ci=0;//mean negative log likelihood

			double t_search  = 0,t_prd = 0;



			Xtst_vec.clear();
			Ytst_vec.clear();
			this->alignTestData_oneStep(groupQuery_vec, Xtst_vec, Ytst_vec, Lvec.back(), ci, y_offset);

			XYtrn_vec.clear();

			//====
			gettimeofday(&tim, NULL);
			double t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
			dataEngine->retrieveTopk(
						Kvec.back(),
						y_offset,
						XYtrn_vec);

			gettimeofday(&tim, NULL);
			double t_end =  tim.tv_sec + (tim.tv_usec / 1000000.0);
			t_search+=t_end-t_start;
			//====
			gettimeofday(&tim, NULL);
			t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
			for (int j = 0; j < groupQuery_vec.size(); j++) {

				if (weightedTrn) {
					prd_vec[j]->makePrediction(Xtst_vec[j], XYtrn_vec[j].Xtrn,
							XYtrn_vec[j].Ytrn, XYtrn_vec[j].dist); //
				} else {
					prd_vec[j]->makePrediction(Xtst_vec[j], XYtrn_vec[j].Xtrn,
							XYtrn_vec[j].Ytrn); //
				}

				double mean, var;

				prd_vec[j]->getPredictonResult(mean, var);

				//for self correction
				int prdRes_set_idx = ci%(prdRes_set[j].size());
				prd_vec[j]->getPredictionMatrix(prdRes_set[j][prdRes_set_idx].meanMat,prdRes_set[j][prdRes_set_idx].varMat,prdRes_set[j][prdRes_set_idx].weightMat);
				prdRes_set[j][prdRes_set_idx].tst = Ytst_vec[j];



				if (selfCorr) {

					//for exp
					if (display){
						print_WeightPrdMat( prd_vec,Ytst_vec,mean,var,ci);//for exp
					}

					//prd_vec[j]->selfCorrWeightPrdMat(Ytst_vec[j]);//for immediately selfCorr//
					//end for exp
					if(ci>prdRes_set[j].size()){
					int corrIdx = (ci%(prdRes_set[j].size()) +prdRes_set[j].size() - y_offset)%(prdRes_set[j].size());
					prd_vec[j]->selfCorrWeightPrdMat(prdRes_set[j][corrIdx].tst,prdRes_set[j][corrIdx].meanMat,prdRes_set[j][corrIdx].varMat);
					}

				}

				double ae = 0, se = 0, sse = 0, nll = 0;

				if (predErr) {

					ae = this->absoluteError(Ytst_vec[j], mean);
					mae_ci += ae;
					se = this->squareError(Ytst_vec[j], mean);
					mse_ci += se;
					sse = this->standSquareError(Ytst_vec[j], mean, var);
					msse_ci += sse;
					nll = this->negLogLikelihood(Ytst_vec[j], mean, var);
					mnll_ci += nll;
				}

				if (display) {
					print_TrnTstData( Xtst_vec,  Ytst_vec, mean,  var,  ae,  se,  sse,  nll);//for exp

				}

			}
			gettimeofday(&tim, NULL);
			t_end =  tim.tv_sec + (tim.tv_usec / 1000000.0);
			t_prd+=t_end-t_start;
			mae+=mae_ci; mse+=mse_ci; msse+=msse_ci; mnll+=mnll_ci;

			//if (predErr) {
			if (display) {

				print_prdPerformancePerStep( ci, mae_ci,t_search, t_prd,  mse_ci,  msse_ci, mnll_ci, groupQuery_vec.size());
			}

			t_search_total+=t_search;
			t_prd_total+=t_prd;
		}

		if (predErr) {



			print_prdPerformance(groupQuery_vec.size(), duration, t_search_total,  t_prd_total,
						 mae,  mse,  msse,  mnll);

		}


		for (int j = 0; j < groupQuery_vec.size(); j++) {
			delete prd_vec[j];
			prd_vec[j] = NULL;
		}

		prd_vec.clear();
		cout<<"successful for TSPred_continuous"<<endl;


	}


	void print_WeightPrdMat(const vector<Predictor<float>*>& prd_vec,const vector<float>& Ytst_vec,double mean, double var, int ci){
		int j=0;//select print sensor id
		//if (j == 0) {
			cout << "for exp print WeightPrdMat for sensor id:" << j
					<< " Ytest:" << Ytst_vec[j] << " mean:" << mean << " var:"
					<< var << " when make prediction step:" << ci << endl;
			vector<vector<double> >& wpm = prd_vec[j]->getWeightPrdMat();
			vector<vector<int> >& wsr =
					prd_vec[j]->getWeightPrdMat_sleepRecord();
			vector<vector<int> >& wpr = prd_vec[j]->getWeightPrdMat_prdRecord();
			vector<vector<int> >& wrs =
					prd_vec[j]->getWeightPrdMat_recoverStep();

			for (int l = 0; l < Lvec.size(); l++) {
				for (int k = 0; k < Kvec.size(); k++) {
					//cerr<<" ("<<l<<","<<k<<") w:"<<wpm[l][k]<<" s:"<<wsr[l][k];

					printf("   (l %d, k %d) w:%.3f s:%d p:%d r:%d pred_mean:%f",
							Lvec[l], Kvec[k], wpm[l][k], wsr[l][k], wpr[l][k],
							wrs[l][k], prd_vec[j]->mean[l][k](0));
				}
				printf("\n");
				//cerr<<endl;
			}
		//}
	}

	void print_TrnTstData(vector<vector<float> >& Xtst_vec, vector<float>& Ytst_vec,double mean, double var, double ae, double se, double sse, double nll){
		int j = 0;
		//if (j == 156) { //for exp

			cout << " Xtst[" << j << "] true observation:" << Ytst_vec[j]
					<< " mean:" << mean << " var:" << var << " absolute error:"
					<< ae << " square error:" << se << " stand Square Error:"
					<< sse << " negative Log Likelihood:" << nll << endl;
		//}

	}

	void print_prdPerformancePerStep(int ci,double t_search, double t_prd, double mae_ci, double mse_ci, double msse_ci,double mnll_ci, int query_vec_size){
		cout << "prediction performance of continuous step: "<<ci << endl
										<< " search time (sec):"<<t_search<<" prediction time (s):"<<t_prd<< " total time(sec):"<<t_search+t_prd<<endl
										<< " mean absolute error:" 	<< mae_ci/(query_vec_size)<<endl
										<< " mean square error:" << mse_ci/(query_vec_size)<<endl
										<< " mean stand Square Error:" << msse_ci/(query_vec_size)<<endl
										<< " mean negative Log Likelihood:"<< mnll_ci/(query_vec_size)<<endl;
	}



	void print_prdPerformance(int query_vec_size,int duration,double t_search_total, double t_prd_total,
			double mae, double mse, double msse, double mnll){

		cout<<" result of prediction (number query:"<<query_vec_size<<" predictio step:"<<duration<<")"<<endl;
					cout<<" time for this step:"<<endl
							<< " search time (sec):"<<t_search_total/duration<<" prediction time (s):"<<t_prd_total/duration<< " total time(sec):"<<(t_search_total+t_prd_total)/duration<<endl;


					cout << "prediction performance:" << endl << " mean absolute error:"
							<< mae/(duration*query_vec_size) << endl << " mean square error:"
							<< mse/(duration*query_vec_size) << endl
							<< " mean stand Square Error:" << msse/(duration*query_vec_size)<< endl
							<< " mean negative Log Likelihood:"<< mnll/(duration*query_vec_size) << endl;
	}



	void set_hypRange(double hypRange){this->hypRange=hypRange;}
	void set_hypSill(double sill){this->hypSill=sill;}
	void set_hypNugget(double nugget){this->hypNugget=nugget;}

	void set_hypPara(double range, double sill, double nugget){
		set_hypRange(range);
		set_hypSill(sill);
		set_hypNugget(nugget);
	}
	void setIgnoreStep(int ignoreStep){
		dataEngine->setIgnoreStep(ignoreStep);
	}

	/**
		 * 	enhancedLowerBound_sel:
	 *						0: use d2q
	 *						1: use q2d
	 *						2: use max(d2q,q2d)
	 */
	void setEnhancedLowerbound(int sel){
		dataEngine->setEnhancedLowerbound(sel);
	}


private:

private:
	//Predictor<int>* predictor;
	//for prediction part
	vector<int> Lvec;
	vector<int> Kvec;

	double hypRange ;
	double hypSill ;
	double hypNugget ;





	DataEngineSmiler* dataEngine;
	int sc_band;
	int winDim;

	bool display;


};

#endif /* TSMANAGER_H_ */
