/*
 * WrapperTimeSeries.h
 *
 *  Created on: May 8, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERTIMESERIES_H_
#define WRAPPERTIMESERIES_H_

#include <string>

using namespace std;

class WrapperTSProcess {
public:
	WrapperTSProcess();
	virtual ~WrapperTSProcess();


	void depressed_runDTW_groupQuery_OnDodgers();
	void runDTW_groupQuery_onDodgers();
	void runDTW_groupQuery_onDodgers_continuousQuery();

	void runDTW_contQuery(string dataHolder, int fcol_start, int fcol_end,
			int queryNumPerBlade,int query_item_len, int contStep,int topk,
			 int sc_band, int windowDim, int enhancedLowerBound_sel=2);
	void runDTW_contGroupQuery(string dataHolder, int fcol_start, int fcol_end,
			int queryNumPerBlade,int* gq_dim_vec, int gq_item_num, int contStep,int topk,
			 int sc_band, int windowDim, int enhancedLowerBound_sel=2 );
	void depressed_runDTW_groupQuery_OnSeqTest();
	void runDTW_groupQuery_onSeqTest();
	void depressed_runDTW_groupQuery_onSeqTest_continuousQuery();
	void runDTW_groupQuery_onSeqTest_continuousQuery();


public:
	/**
	 * TODO:
	 * do topkquery
	 */
	void depressed_Genie_topkQuery(
			string dataHolder, int dataFileCol,//indicate the data file and loading which column
			int bits_for_value, int windowDim, int bladeNum, //parameter for building index: bits_for_value - how many bits to store the value in low bits; windowDim - dimension of window query; bladeNum - number of data blades
			int topk,int groupQueryNum,int* gq_dim_vec, int gq_item_num, int maxQueryLen, //parameter for query:
			int numOfDocToExpand, int sc_band, int upSearchBoundExtreme, int downSearchBoundExtreme//parameter for the configuration of queries
			);

	/**
	 * TODO:
	 * do topkquery
	 */
	void Genie_topkQuery(
			string dataHolder, int dataFileCol,//indicate the data file and loading which column
			int windowDim, int sc_band, int bladeNum, //parameter for building index: windowDim - dimension of window query; bladeNum - number of data blades
			int topk,int groupQueryNum,int* gq_dim_vec, int gq_item_num, int maxQueryLen //parameter for query:
		);

};

#endif /* WRAPPERTIMESERIES_H_ */
