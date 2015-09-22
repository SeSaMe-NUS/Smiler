/*
 * WrapperTSManager.cpp
 *
 *  Created on: Jun 26, 2014
 *      Author: zhoujingbo
 */

#include "WrapperTSManagerOnGpuIdx.h"

#include <vector>
#include <string>



#include "TSManagerOnGpuIdx.h"
#include "../searcher/UtlGenie.h"


#include <sstream>

WrapperTSManagerOnGpuIdx::WrapperTSManagerOnGpuIdx() {
	// TODO Auto-generated constructor stub

}

WrapperTSManagerOnGpuIdx::~WrapperTSManagerOnGpuIdx() {
	// TODO Auto-generated destructor stub
}



/**
 * use the first Lvec[max] as the test data to find topk, the query with larger than Lvec[max] is ignored
 */

void WrapperTSManagerOnGpuIdx::alignQuery(vector<vector<float> >& query_vec,
			vector<vector<float> >& _Xtst_vec, vector<int>& Lvec) {

		//align the query with Lvec
		_Xtst_vec.resize(query_vec.size());
		for (int i = 0; i < _Xtst_vec.size(); i++) {
			_Xtst_vec[i].resize(Lvec.back());

			std::copy(query_vec[i].begin(),
					query_vec[i].begin() + Lvec.back(),
					_Xtst_vec[i].begin());

		}
	}





//=========================== key function========================


void WrapperTSManagerOnGpuIdx::runMain_TSLOOCVContPred_mulSensors(
		string fileHolder,
		int fcol_start, int fcol_end, int queryNumPerBlade,
		int contPrdStep, int queryLenLoaded,
		vector<int>& Lvec, vector<int>& Kvec, double range, double sill, double nugget,
		int y_offset, int sc_band, bool selfCorr, bool weightedTrn
	){

	runMain_TSContPred_mulSensors(
			fileHolder,
			fcol_start, fcol_end, queryNumPerBlade,
			contPrdStep,queryLenLoaded,
			Lvec, Kvec, range, sill, nugget,
			y_offset, sc_band, 0,  selfCorr,  weightedTrn
		);
}


void WrapperTSManagerOnGpuIdx::runMain_TSKnnRegContPred_mulSensors(
		string fileHolder,
		int fcol_start, int fcol_end, int queryNumPerBlade,
		int contPrdStep,  int queryLenLoaded,
		vector<int>& Lvec, vector<int>& Kvec,
		int y_offset, int sc_band, bool selfCorr, bool weightedTrn
	){

	runMain_TSContPred_mulSensors(
			fileHolder,
			fcol_start, fcol_end, queryNumPerBlade,
			contPrdStep,queryLenLoaded,
			Lvec, Kvec, 1, 1, 1,
			y_offset, sc_band, 1,  selfCorr,  weightedTrn
		);
}



/**
 * pred_selector:
 * 			   0: LOOCV predictor
 * 			   1: KnnReg predictor
 */
void WrapperTSManagerOnGpuIdx::runMain_TSContPred_mulSensors(
		string fileHolder,
		int fcol_start, int fcol_end, int queryNumPerBlade,
		int contPrdStep, int queryLenLoaded,
		vector<int>& Lvec, vector<int>& Kvec, double range, double sill, double nugget,
		int y_offset, int sc_band, int pred_sel, bool selfCorr, bool weightedTrn
	){

	int bladeNum =  fcol_end-fcol_start+1;
	//configuration for query
	int queryItemMaxLen = Lvec.back()+y_offset;
	int maxQueryLen = queryItemMaxLen+queryLenLoaded;
	int groupQueryNum = bladeNum*queryNumPerBlade;

	vector<vector<float> > query_master_vec;
	vector<int> query_blade_map;
	vector<vector<float> > bladeData_vec;

	UtlGenie::loadDataQuery_leaveOut_mulBlades( fileHolder,  fcol_start,  fcol_end,  queryNumPerBlade,
			maxQueryLen,  contPrdStep,
			//queryItemMaxLen,  contPrdStep,//for sigmod1stSubmit branch
			query_master_vec,
		    query_blade_map,
			bladeData_vec
			);

	TSManagerOnGpuIdx tsManager;
	tsManager.setDisplay(false);


	tsManager.setIgnoreStep(0);
	tsManager.setEnhancedLowerbound(2);

	tsManager.conf_DataEngine(bladeData_vec, sc_band,2*sc_band);
	tsManager.conf_Predictor(Lvec, Kvec);
	tsManager.set_hypPara( range,  sill,  nugget);

	tsManager.TSPred_continuous_onGPUIdx(query_master_vec, contPrdStep, query_blade_map,
				y_offset, pred_sel, weightedTrn, selfCorr);
}



/**
 * TODO:
 *     key function for  continuous (y_offset+1) step ahead  prediction wrapper
 */
void WrapperTSManagerOnGpuIdx::runMain_TSPred_continuous(vector<vector<float> >& blade_data_vec,
		vector<int>& query_blade_map, vector<vector<float> >& query_vec,
		int contPrdStep,
		vector<int>& Lvec, vector<int>& Kvec, double range, double sill, double nugget,
		int y_offset, int sc_band, int pred_sel,bool weightedTrn, bool selfCorr, bool ignoreFirstStep){

	TSManagerOnGpuIdx tsManager;
	tsManager.setDisplay(false);

	if(ignoreFirstStep){
		tsManager.setIgnoreStep(1);
	}else{
		tsManager.setIgnoreStep(0);
	}

	tsManager.conf_DataEngine(blade_data_vec, sc_band, sc_band*2);//set the default winDim as two times of sc_band
	tsManager.conf_Predictor(Lvec, Kvec);
	tsManager.set_hypPara( range,  sill,  nugget);

	tsManager.TSPred_continuous_onGPUIdx(query_vec, contPrdStep, query_blade_map,
			y_offset, pred_sel, weightedTrn, selfCorr);

}


//============================


//================for continuous prediction=================


void WrapperTSManagerOnGpuIdx::runExp_seq_TSPred_continuous(string dataFile, int columnAv, int queryNum, int queryLen, int contPrdStep, vector<int>& Lvec, vector<int>& Kvec,
		double range, double sill, double nugget, int y_offset, int sc_band, int pred_sel,bool weighted, bool selfCorr){
	vector<float> loadData;
	DataProcess dp;
	dp.ReadFileFloat(dataFile.c_str(), columnAv, loadData);

	QuerySampler qser;
	vector<vector<float> > query_vec;

	int seq_start_pos = loadData.size() - queryNum - queryLen;

	qser.getSeqQuery(loadData, queryNum, queryLen, query_vec, seq_start_pos);
	vector<int> query_blade_map(query_vec.size(), 0);
	vector<vector<float> > blade_data_vec(1);
	blade_data_vec[0].resize(loadData.size() - queryNum);	//for exp
	std::copy(loadData.begin(), loadData.begin() + seq_start_pos,
			blade_data_vec[0].begin());


	bool ignoreFirstStep = false;
	runMain_TSPred_continuous( blade_data_vec,  query_blade_map,  query_vec, contPrdStep,
			 Lvec,  Kvec,  range,  sill,  nugget,
			 y_offset,  sc_band,  pred_sel, weighted,  selfCorr, ignoreFirstStep);
}

void WrapperTSManagerOnGpuIdx::runExp_seq_TSPred_continuous(string fileHolder,
		int dataFileCol,  int queryNum, int queryLen, int contPrdStep, int dimensionNum, int topk, double range,
		double sill, double nugget, int y_offset, int sc_band, int pred_sel,
		bool weighted, bool selfCorr){


	string dataFile = UtlGenie::getDataFile(fileHolder);
	cout << "data file:" << dataFile << endl;
	vector<int> Lvec(1, dimensionNum);
	vector<int> Kvec(1, topk);


	runExp_seq_TSPred_continuous(
			dataFile, dataFileCol,
			queryNum, queryLen, contPrdStep,
			Lvec, Kvec,
			range, sill, nugget,
			y_offset,
			sc_band, pred_sel,
			weighted, selfCorr);

}


//================end for continuous prediction===============

//========================================================

//========================================================


//=============================================================================

void WrapperTSManagerOnGpuIdx::getSeqQueryAndData(string dataFile, int columnAv,
		int queryNum, int queryLen,
		vector<vector<float> >& _blade_data_vec,
		vector<vector<float> >& _query_vec,vector<int>& _query_blade_map){

	vector<float> loadData;
	DataProcess dp;
	dp.ReadFileFloat(dataFile.c_str(), columnAv, loadData);

	QuerySampler qser;

	int seq_start_pos = loadData.size() - queryNum - queryLen;
	qser.getSeqQuery(loadData, queryNum, queryLen, _query_vec, seq_start_pos);

	_query_blade_map.resize(_query_vec.size(),0);
	_blade_data_vec.resize(1);
	_blade_data_vec[0].resize(loadData.size() - queryNum);	//for exp
	std::copy(loadData.begin(), loadData.begin() + seq_start_pos,
				_blade_data_vec[0].begin());



}


//===================================================================================
