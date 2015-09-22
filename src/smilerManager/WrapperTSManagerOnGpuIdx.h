/*
 * WrapperTSManager.h
 *
 *  Created on: Jun 26, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERTSMANAGERONGPUINDEX_H_
#define WRAPPERTSMANAGERONGPUINDEX_H_

#include <vector>
#include <string>
#include "../searcher/UtlGenie.h"
using namespace std;


class WrapperTSManagerOnGpuIdx {
public:
	WrapperTSManagerOnGpuIdx();
	virtual ~WrapperTSManagerOnGpuIdx();


	//================ ===================


	void runMain_TSPred_continuous(vector<vector<float> >& blade_data_vec, vector<int>& query_blade_map, vector<vector<float> >& query_vec, int contPrdStep,
			vector<int>& Lvec, vector<int>& Kvec, double range, double sill, double nugget,
			int y_offset, int sc_band, int pred_sel,bool weightedTrn, bool selfCorr, bool ignoreFirstStep);



	void runMain_TSLOOCVContPred_mulSensors(
			string fileHolder,
			int fcol_start, int fcol_end, int queryNumPerBlade,
			int contPrdStep, int queryLenLoaded,
			vector<int>& Lvec, vector<int>& Kvec, double range, double sill, double nugget,
			int y_offset, int sc_band, bool selfCorr, bool weightedTrn=false
		);

	void runMain_TSKnnRegContPred_mulSensors(
			string fileHolder,
			int fcol_start, int fcol_end, int queryNumPerBlade,
			int contPrdStep, int queryLenLoaded,
			vector<int>& Lvec, vector<int>& Kvec,
			int y_offset, int sc_band, bool selfCorr, bool weightedTrn=false
		);

	void runMain_TSContPred_mulSensors(
			string fileHolder,
			int fcol_start, int fcol_end, int queryNumPerBlade,
			int contPrdStep, int queryLenLoaded,
			vector<int>& Lvec, vector<int>& Kvec, double range, double sill, double nugget,
			int y_offset, int sc_band, int pred_sel, bool selfCorr, bool weightedTrn=false
		);


	/**
	 *
	 */
	//======================================

	//==============================for continuous prediction==============

	//
	/**
	 * TODO:
	 * for one dimension with one topk
	 */
	void runExp_seq_TSLOOCVPred_continuous(
			string fileHolder,	int dataFileCol,
			int queryNum, int contPrdStep,
			int dimensionNum, int topk,
			double range,double sill, double nugget,
			int y_offset, int sc_band,
			bool weighted, bool selfCorr){

		int queryLen = dimensionNum+y_offset+contPrdStep;

		runExp_seq_TSPred_continuous( fileHolder,
					 dataFileCol,   queryNum,  queryLen, contPrdStep, dimensionNum,  topk,  range,
					 sill,  nugget,  y_offset,  sc_band,  0,
					 weighted,  selfCorr);
	}


	/**
	 * TODO:
	 * 		prediction with multiple dimensions and topks
	 */
	void runExp_seq_TSLOOCVPred_continuous(
			string fileHolder, int dataFileCol,
			int queryNum, int contPrdStep,
			vector<int>& Lvec, vector<int>& Kvec,
			double range,double sill, double nugget,
			int y_offset, int sc_band, bool weighted, bool selfCorr){

		int queryLen = Lvec.back()+y_offset+contPrdStep;
		string dataFile = UtlGenie::getDataFile(fileHolder);
		cout << "data file:" << dataFile << endl;

		runExp_seq_TSPred_continuous(
					dataFile, dataFileCol,
					queryNum, queryLen, contPrdStep,
					Lvec, Kvec,
					range, sill, nugget,
					y_offset,
					sc_band, 0,
					weighted, selfCorr);
	}

	/**
	 * TODO:
	 * prediction with one dimension and topk
	 */
	void runExp_seq_TSKnnRegPred_continuous(string fileHolder,
			int dataFileCol,  int queryNum, int contPrdStep, int dimensionNum, int topk,
			int y_offset, int sc_band,
			bool weighted, bool selfCorr){

		int queryLen = dimensionNum+y_offset+contPrdStep;

		runExp_seq_TSPred_continuous( fileHolder,
							 dataFileCol,   queryNum,  queryLen, contPrdStep, dimensionNum,  topk,
							 0,	 0,  0,  y_offset,  sc_band,  1,
							 weighted,  selfCorr);

	}

	void runExp_seq_TSKnnRegPred_continuous(string fileHolder, int dataFileCol,
			int queryNum, int contPrdStep,
			vector<int>& Lvec,vector<int>& Kvec,
			int y_offset,int sc_band,
			bool weighted, bool selfCorr){

		int queryLen = Lvec.back()+y_offset+contPrdStep;
		string dataFile = UtlGenie::getDataFile(fileHolder);
		cout << "data file:" << dataFile << endl;

		runExp_seq_TSPred_continuous(
							dataFile, dataFileCol,
							queryNum, queryLen, contPrdStep,
							Lvec, Kvec,
							0, 0, 0,
							y_offset,
							sc_band, 1,
							weighted, selfCorr);


	}

	void runExp_seq_TSPred_continuous(string fileHolder,
			int dataFileCol,  int queryNum, int queryLen, int contPrdStep, int dimensionNum, int topk, double range,
			double sill, double nugget, int y_offset, int sc_band, int pred_sel,
			bool weighted, bool selfCorr);

	void runExp_seq_TSPred_continuous(string dataFile, int columnAv, int queryNum, int queryLen, int contPrdStep, vector<int>& Lvec, vector<int>& Kvec,
			double range, double sill, double nugget, int y_offset, int sc_band, int pred_sel,bool weighted, bool selfCorr);

	//=================================end for continuous prediction===============


	void runExp_TSPred_itrMulStep(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk,
			double range, double sill, double nugget, int mulStep, int sc_band, int pred_sel,bool weighted);

	void runExp_TSLOOCVPred_itrMulStep(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk,
				double range, double sill, double nugget, int mulStep, int sc_band,bool weighted){
		runExp_TSPred_itrMulStep( fileHolder,  dataFileCol,  dimensionNum,  queryNum,  topk,
					 range,  sill,  nugget,  mulStep,  sc_band,  0, weighted);
	}

	void runExp_TSKnnRegPred_itrMulStep(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk,
					 int mulStep, int sc_band,bool weighted=false){
			runExp_TSPred_itrMulStep( fileHolder,  dataFileCol,  dimensionNum,  queryNum,  topk,
						 0,  0,  0,  mulStep,  sc_band,  1, weighted);
		}

	void run_TSLOOCVPred_itrMulStep();




	//experiment, the query is random selected
	void runExp_TSPred_oneStep(string fileHolder, int dataFileCol,
			int dimensionNum, int queryNum, int topk,
			double range, double sill, double nugget,
			int y_offset,
			int sc_band, int pred_sel,bool weighted);

	void runExp_TSLOOCVPred_oneStep(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk,
			double range, double sill, double nugget,
			int y_offset,
			int sc_band, bool weighted){
		runExp_TSPred_oneStep( fileHolder,  dataFileCol,  dimensionNum,  queryNum,  topk, range,  sill,  nugget,y_offset, sc_band, 0, weighted);
	}

	void runExp_TSKnnRegPred_oneStep(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk,
			int y_offset,
			int sc_band, bool weighted){
		runExp_TSPred_oneStep( fileHolder,  dataFileCol,  dimensionNum,  queryNum,  topk,   0, 0, 0, y_offset, sc_band,  1,weighted);
	}



	//==================================================competitor================================
	void runExp_psgp_trnAndPrd_mulSensors(string fileHolder,
			int fcol_start, int fcol_end,
			int queryNum,
			int dim, int activePointsNum,
			int y_offset);


private:
	void getSeqQueryAndData(string dataFile, int columnAv,
			int queryNum, int queryLen,
			vector<vector<float> >& _blade_data_vec,
			vector<vector<float> >& _query_vec,vector<int>& _query_blade_map);

	void alignQuery(vector<vector<float> >& query_vec,
				vector<vector<float> >& _Xtst_vec, vector<int>& Lvec);

};

#endif /* WRAPPERTSMANAGER_H_ */
