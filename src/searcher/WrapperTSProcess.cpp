/*
 * WrapperTimeSeries.cpp
 *
 *  Created on: May 8, 2014
 *      Author: zhoujingbo
 */

#include "WrapperTSProcess.h"

#include <sstream>
#include <iostream>
#include <sys/time.h>
using namespace std;

#include "GPUKNN/UtlGPU.h"
#include "GPUKNN/generalization.h"
#include "GPUKNN/GPUManager.h"
#include "TSProcess/TSProcessor.h"
#include "../tools/DataOperator/DataProcess.h"
#include "../tools/BladeLoader.h"
#include "../tools/DataOperator/QuerySampler.h"
#include "UtlGenie.h"



template <class T>
void setup_pesudoGroupQuries(vector<vector<T> > groupQueryData, int* gq_dim_vec, int gq_item_num, int winDim,
		vector<GroupQuery_info*>& gpuQuery_info_set){

	//load data files

	gpuQuery_info_set.clear();
	gpuQuery_info_set.reserve(groupQueryData.size());

	int maxQueryLen = gq_dim_vec[gq_item_num -1];
	int winNumPerGroup = maxQueryLen - winDim +1;



	for(int i=0;i<groupQueryData.size();i++){

		vector<float> gq_data (maxQueryLen);
		for(int j=0;j<maxQueryLen;j++){
			gq_data[j]= groupQueryData[i][j];
		}

		//GroupQuery_info(int groupId, int blade_id, int startQueryId, int* groupQueryDimensions_vec, int groupQuery_item_number,float* gq_data)
		GroupQuery_info* groupQuery_info = new GroupQuery_info(i,i,0,gq_dim_vec,gq_item_num,gq_data.data());

		gpuQuery_info_set.push_back(groupQuery_info);
	}

}


template <class T>
void setup_groupQuery(vector<vector<T> > groupQuery_vec, vector<int> query_blade_map,int* gq_dim_vec, int gq_item_num, int winDim,
		vector<GroupQuery_info*>& gpuQuery_info_set){

	gpuQuery_info_set.clear();
	gpuQuery_info_set.reserve(groupQuery_vec.size());

	int maxQueryLen = gq_dim_vec[gq_item_num - 1];
	int winNumPerGroup = maxQueryLen - winDim + 1;

	for (int i = 0; i < groupQuery_vec.size(); i++) {

		vector<float> gq_data(maxQueryLen);
		for (int j = 0; j < maxQueryLen; j++) {
			gq_data[j] = groupQuery_vec[i][j];
		}

		//GroupQuery_info(int groupId, int blade_id, int startQueryId, int* groupQueryDimensions_vec, int groupQuery_item_number,float* gq_data)
		GroupQuery_info* groupQuery_info = new GroupQuery_info(i, query_blade_map[i], 0,
				gq_dim_vec, gq_item_num, gq_data.data());

		gpuQuery_info_set.push_back(groupQuery_info);
	}

}

void depressed_setup_pesudoBlades(const BladeLoader<int> bldLoader,const int pesudo_bladeNum,
		vector<float>& _pesudo_blade_data_vec,vector<int>& _pesudo_blade_len_vec){

	int blade_item_len = bldLoader.data.size();

	_pesudo_blade_data_vec.resize(pesudo_bladeNum*blade_item_len);
	_pesudo_blade_len_vec.resize(pesudo_bladeNum,blade_item_len);

	for(int i=0;i<pesudo_bladeNum;i++){
		for(int j = 0;j<blade_item_len;j++){
			_pesudo_blade_data_vec[i*blade_item_len+j] = bldLoader.data[j];
		}
	}

}

void setup_pesudoBlades(const BladeLoader<float> bldLoader, const int pesudo_bladeNum,
		vector<float>& _pesudo_blade_data_vec,vector<int>& _pesudo_blade_len_vec){

	int blade_item_len = bldLoader.data.size();

	_pesudo_blade_data_vec.resize(pesudo_bladeNum*blade_item_len);
	_pesudo_blade_len_vec.resize(pesudo_bladeNum,blade_item_len);

	for(int i=0;i<pesudo_bladeNum;i++){
		for(int j = 0;j<blade_item_len;j++){
				_pesudo_blade_data_vec[i*blade_item_len+j] = bldLoader.data[j];
		}
	}
}

void setupBlade( string dataFileHolder, int dataFile_col, TSProcessor<float>& tsp,  int bladeNum){

	string dataFile = dataFileHolder+".csv";
	BladeLoader<float> bldLoader;
	bldLoader.loadDataFloat(dataFile.c_str(),dataFile_col);

	vector<float> blade_data_vec;
	vector<int> blade_len_vec;

	setup_pesudoBlades(bldLoader, bladeNum,
			blade_data_vec,blade_len_vec);

	//const int bladeNum, vector<float>& blade_data_vec, vector<int>& blade_len_vec,const int winDim
	tsp.conf_TSGPUManager_blade(bladeNum,blade_data_vec,blade_len_vec);
}



void depressed_setupOnIntegerData_BladeAndIndex(const string dataHolder, int dataFile_col, TSProcessor<int>& tsp,
		const int numOfDocToExpand, const int bits_for_value, const float bucketWidth,
		const int winDim, const int bladeNum, const int index_totalDimensions){


	string dataFile = dataHolder+".csv";
	//int dataFile_col = 0;

	BladeLoader<int> bldLoader;
	bldLoader.loadDataInt(dataFile.c_str(),dataFile_col);
	vector<float> blade_data_vec;
	vector<int> blade_len_vec;
	depressed_setup_pesudoBlades(bldLoader, bladeNum, blade_data_vec,blade_len_vec);


	std::stringstream idxPath;
	idxPath<<dataHolder<<"_wd"<<winDim<<"_bld"<< bladeNum<<"_bv"<<bits_for_value<<"_dw_group.idx";
	cout<<"idx path is:"<<idxPath.str()<<endl;


	tsp.depressed_conf_TSGPUManager_index_blade(numOfDocToExpand, winDim, idxPath.str(), bits_for_value, bucketWidth,
			bladeNum,blade_data_vec,blade_len_vec);

}

void setupQuery(const string dataHolder,TSProcessor<float>& tsp, const int queryLen, const int groupQueryNum,
		const int winDim, const int sc_band,
		int* gq_dim_vec, const int gq_item_num){

	std::stringstream queryFileStream;
	queryFileStream << dataHolder << "_ql" << queryLen << "_gqn"
			<< groupQueryNum << "_group.query";
	string queryFile = queryFileStream.str();
	cout << "query file is:" << queryFile << endl;
	vector<GroupQuery_info*> groupQuery_info_Set; //this vector will be released in the TSGPUManager

	//load data files
	DataProcess dp;
	vector<vector<float> > groupQueryData;
	dp.ReadFileFloat(queryFile.c_str(), groupQueryData);

	setup_pesudoGroupQuries(groupQueryData, gq_dim_vec, gq_item_num, winDim,
							groupQuery_info_Set);

	//int sc_band,int queryLen, int itemNum_perGroup, vector<GroupQuery_info*>& groupQuery_info_set
	tsp.conf_TSGPUManager_query(winDim,sc_band, gq_dim_vec[gq_item_num-1],gq_item_num,groupQuery_info_Set);
}

void depressed_setupOnIntegerData_Query(const string dataHolder,TSProcessor<int>& tsp, const int queryLen, const int groupQueryNum, const int sc_band,
		int* gq_dim_vec, const int gq_item_num, const int winDim, const int upSearchBoundExtreme, const int downSearchBoundExtreme){

		std::stringstream queryFileStream;
		queryFileStream<<dataHolder<<"_ql"<<queryLen<<"_gqn"<<groupQueryNum<<"_group.query";
		string queryFile= queryFileStream.str();
		cout<<"query file is:"<<queryFile<<endl;
		vector<GroupQuery_info*> groupQuery_info_Set;//this vector will be released in the TSGPUManager

		//load data files
		DataProcess dp;
		vector<vector<int> > groupQueryData;
		dp.ReadFileInt(queryFile.c_str(),groupQueryData);

		vector<int> upSearchBoundExtreme_vec(groupQueryData.size(),upSearchBoundExtreme);
		vector<int> downSearchBoundExtreme_vec(groupQueryData.size(),downSearchBoundExtreme);
		setup_pesudoGroupQuries(groupQueryData, gq_dim_vec, gq_item_num, winDim,
						groupQuery_info_Set);

		tsp.depressed_conf_TSGPUManager_query(sc_band, gq_dim_vec[gq_item_num-1],gq_item_num,
				groupQuery_info_Set, upSearchBoundExtreme_vec, downSearchBoundExtreme_vec );
}

void depressed_setupOnSeqTest_continuousQuery(const string dataHolder,TSProcessor<int>& tsp, const int queryLenLoaded, const int groupQueryNum, const int sc_band,
		int* gq_dim_vec, const int gq_item_num, const int winDim, const int upSearchBoundExtreme, const int downSearchBoundExtreme){



	std::stringstream queryFileStream;
	queryFileStream<<dataHolder<<"_ql"<<queryLenLoaded<<"_gqn"<<groupQueryNum<<"_group.query";
	string queryFile= queryFileStream.str();
	cout<<"query file is:"<<queryFile<<endl;
	vector<GroupQuery_info*> groupQuery_info_Set;//this vector will be released in the TSGPUManager
	vector<GpuWindowQuery> windowQuerySet;
	//setup_pesudoGroupQuries(queryFile, gq_dim_vec,gq_item_num, groupQuery_info_Set);

	//load data files
	DataProcess dp;
	vector<vector<int> > groupQueryData;
	dp.ReadFileInt(queryFile.c_str(),groupQueryData);

	vector<int> upSearchBoundExtreme_vec(groupQueryData.size(),upSearchBoundExtreme);
	vector<int> downSearchBoundExtreme_vec(groupQueryData.size(),downSearchBoundExtreme);

	setup_pesudoGroupQuries(groupQueryData, gq_dim_vec, gq_item_num, winDim,
			groupQuery_info_Set);


	tsp.depressed_conf_TSGPUManager_continuousQuery(sc_band, gq_dim_vec[gq_item_num-1],gq_item_num,
			groupQuery_info_Set, groupQueryData, upSearchBoundExtreme_vec,downSearchBoundExtreme_vec);
}



void setupOnSeqTest_continuousQuery(const string dataHolder,TSProcessor<float>& tsp, const int queryLenLoaded, const int groupQueryNum, const int sc_band,
		int* gq_dim_vec, const int gq_item_num, const int winDim){

	std::stringstream queryFileStream;
	queryFileStream<<dataHolder<<"_ql"<<queryLenLoaded<<"_gqn"<<groupQueryNum<<"_group.query";
	string queryFile= queryFileStream.str();
	cout<<"query file is:"<<queryFile<<endl;
	vector<GroupQuery_info*> groupQuery_info_Set;//this vector will be released in the TSGPUManager
	vector<GpuWindowQuery> windowQuerySet;
	//setup_pesudoGroupQuries(queryFile, gq_dim_vec,gq_item_num, groupQuery_info_Set);

	//load data files
	DataProcess dp;
	vector<vector<float> > groupQueryData;
	dp.ReadFileFloat(queryFile.c_str(),groupQueryData);

	setup_pesudoGroupQuries(groupQueryData, gq_dim_vec, gq_item_num, winDim, groupQuery_info_Set);

	tsp.conf_TSGPUManager_continuousQuery(winDim,sc_band, gq_dim_vec[gq_item_num-1],gq_item_num,
			groupQuery_info_Set, groupQueryData);

}

void setup_contGroupQuery(TSProcessor<float>& tsp,
		vector<vector<float> > groupQuery_vec, vector<int> query_blade_map, int* gq_dim_vec, const int gq_item_num,
		const int sc_band, const int winDim){

	vector<GroupQuery_info*> groupQuery_info_Set;//this vector will be released in the TSGPUManager

	setup_groupQuery(groupQuery_vec, query_blade_map, gq_dim_vec, gq_item_num,  winDim,
			groupQuery_info_Set);

	tsp.conf_TSGPUManager_continuousQuery(winDim,sc_band, gq_dim_vec[gq_item_num-1],gq_item_num,
				groupQuery_info_Set, groupQuery_vec);
}


WrapperTSProcess::WrapperTSProcess() {
	// TODO Auto-generated constructor stub

}

WrapperTSProcess::~WrapperTSProcess() {
	// TODO Auto-generated destructor stub
}


void WrapperTSProcess::runDTW_groupQuery_onSeqTest(){

		string dataHolder = "data/test/sequenceTest_temp";
		int dataFileCol = 0;
		//string dataHolder = "data/test/seqTest_8k";


		cout << "Starting: WrapperTSProcess::runDTW_groupQuery_onSeqTest"<< endl;

		//configuration for blade and index
		int windowDim = 4;
		int sc_band = windowDim/2;

		//configuration for query
		int bladeNum = 2;
		int topk = 3;//3;
		int maxQueryLen = 16;
		int groupQueryNum = bladeNum;
		int gq_dim_vec[3] = {8,12,16};
		int gq_item_num = 3;

		//Genie_topkQuery(
		//string dataHolder, int dataFileCol,//indicate the data file and loading which column
		//int windowDim, int sc_band, int bladeNum, //parameter for building index: windowDim - dimension of window query; bladeNum - number of data blades
		//int topk,int groupQueryNum,int* gq_dim_vec, int gq_item_num, int maxQueryLen //parameter for query:
		//)
		this->Genie_topkQuery(dataHolder, dataFileCol,//indicate the data file and loading which column
				 windowDim, sc_band, bladeNum, //parameter for building index: bits_for_value - how many bits to store the value in low bits; windowDim - dimension of window query; bladeNum - number of data blades
				topk,groupQueryNum, gq_dim_vec,  gq_item_num,  maxQueryLen //parameter for query:
				 );


}

/**
 * TODO:
 * configure the index and query for direct query
 */
void WrapperTSProcess::depressed_runDTW_groupQuery_OnSeqTest(){

	string dataHolder = "data/test/sequenceTest";
	int dataFileCol = 0;
	//string dataHolder = "data/test/seqTest_8k";


	cout << "Starting: WrapperTSProcess::runDTW_groupQuery_OnSequenceTest"<< endl;

	//configuration for blade and index
	int bits_for_value = 7;
	float bucketWith = 1.;
	int windowDim = 4;
	int bladeNum = 2;
	int index_totalDimensions = bladeNum*windowDim;


	//configuration for query
	int topk = 3;//3;
	int numOfDocToExpand = 10;
	int sc_band = windowDim/2;
	int maxQueryLen = 16;
	int groupQueryNum = bladeNum;
	int gq_dim_vec[3] = {8,12,16};
	int gq_item_num = 3;
	int upSearchBoundExtreme = 127;
	int downSearchBoundExtreme = 0;

	depressed_Genie_topkQuery(
			dataHolder, dataFileCol,//indicate the data file and loading which column
			bits_for_value, windowDim, bladeNum, //parameter for building index: bits_for_value - how many bits to store the value in low bits; windowDim - dimension of window query; bladeNum - number of data blades
			topk,groupQueryNum, gq_dim_vec,  gq_item_num,  maxQueryLen, //parameter for query:
			 numOfDocToExpand,  sc_band,  upSearchBoundExtreme,  downSearchBoundExtreme//parameter for the configuration of queries
			);
}


void  WrapperTSProcess::depressed_runDTW_groupQuery_onSeqTest_continuousQuery(){

	string dataHolder = "data/test/sequenceTest";
	int fcol = 0;
	//string dataHolder = "data/test/seqTest_8k";

	cout << "Starting: WrapperTSProcess::depressed_runDTW_groupQuery_onSeqTest_continuousQuery()"<< endl;

	//configuration for blade and index
	int bits_for_value = 7;
	float bucketWidth = 1;
	int winDim = 4;
	int bladeNum = 2;
	int index_totalDimensions = bladeNum*winDim;

	//configuration for query
	int topk = 3;//3;
	int numOfDocToExpand = 10;
	int sc_band = winDim/2;
	int queryLenLoaded = 32;
	int queryLen = 16;
	int groupQueryNum = bladeNum;
	int gq_dim_vec[3] = {8,12,16};
	int gq_item_num = 3;
	int upSearchBoundExtreme = 127;
	int downSearchBoundExtreme = 0;

	TSProcessor<int> tsp;
	depressed_setupOnIntegerData_BladeAndIndex(dataHolder,fcol, tsp, numOfDocToExpand, bits_for_value, bucketWidth,  winDim, bladeNum, index_totalDimensions);

	depressed_setupOnSeqTest_continuousQuery(dataHolder,tsp,  queryLenLoaded, groupQueryNum,  sc_band,
			 gq_dim_vec,  gq_item_num,  winDim,  upSearchBoundExtreme,  downSearchBoundExtreme);

	tsp.depressed_continous_topkQuery_dtw(topk,queryLenLoaded-queryLen);

}


void  WrapperTSProcess::runDTW_groupQuery_onSeqTest_continuousQuery(){

	string dataHolder = "data/test/sequenceTest";
	int fcol = 0;
	//string dataHolder = "data/test/seqTest_8k";

	cout << "Starting: WrapperTSProcess::runDTW_groupQuery_onSeqTest_continuousQuery()"<< endl;

	//configuration for blade and index
	int winDim = 4;
	int bladeNum = 2;

	//configuration for query
	int topk = 3;//3;
	int sc_band = winDim/2;
	int queryLenLoaded = 32;
	int queryLen = 16;
	int groupQueryNum = bladeNum;
	int gq_dim_vec[3] = {8,12,16};
	int gq_item_num = 3;

	TSProcessor<float> tsp;
	//depressed_setupOnIntegerData_BladeAndIndex(dataHolder,fcol, tsp, numOfDocToExpand, bits_for_value, bucketWidth,  winDim, bladeNum, index_totalDimensions);

	setupBlade(dataHolder,  fcol,  tsp, bladeNum);;

	setupOnSeqTest_continuousQuery(dataHolder,tsp,  queryLenLoaded, groupQueryNum,  sc_band,
			 gq_dim_vec,  gq_item_num,  winDim);

	tsp.continous_topkQuery_dtw(topk,queryLenLoaded-queryLen);

}

/**
 * select the mode for enhancedLowerBound, 	enhancedLowerBound_sel: 0: use d2q, 1: use q2d, 2: use max(d2q,q2d)
 */
void  WrapperTSProcess::runDTW_contQuery(string dataHolder, int fcol_start, int fcol_end,
		int queryNumPerBlade,int query_item_len, int contStep,int topk,
		 int sc_band, int windowDim, int enhancedLowerBound_sel){

	int gq_item_num = 1;
	int * gq_dim_vec=new int[gq_item_num];
	gq_dim_vec[0] = query_item_len;

	runDTW_contGroupQuery( dataHolder,  fcol_start,  fcol_end,
			 queryNumPerBlade, gq_dim_vec,  gq_item_num,  contStep, topk,
			  sc_band,  windowDim,  enhancedLowerBound_sel);

	delete[] gq_dim_vec;
}

/**
 * select the mode for enhancedLowerBound, 	enhancedLowerBound_sel: 0: use d2q, 1: use q2d, 2: use max(d2q,q2d)
 */
void WrapperTSProcess::runDTW_contGroupQuery(string fileHolder, int fcol_start, int fcol_end,
		int queryNumPerBlade,int* gq_dim_vec, int gq_item_num, int contStep,int topk,
		int sc_band, int windowDim, int enhancedLowerBound_sel){

		int bladeNum =  fcol_end-fcol_start+1;
		//configuration for query
		int queryItemMaxLen = gq_dim_vec[gq_item_num-1];
		int queryLenLoaded = queryItemMaxLen+contStep;
		int groupQueryNum = bladeNum*queryNumPerBlade;

		TSProcessor<float> tsp;
		//depressed_setupOnIntegerData_BladeAndIndex(dataHolder,fcol, tsp, numOfDocToExpand, bits_for_value, bucketWidth,  winDim, bladeNum, index_totalDimensions);
		tsp.setEnhancedLowerBound_sel(enhancedLowerBound_sel);
		cout<<" enhancedLowerBound_sel:"<<tsp.getEnhancedLowerBound_sel()<<endl;
		tsp.setEnableSumUnfilteredCandidates(true);//for exp statistics, record the unfiltered candidates

		vector<vector<float> > query_master_vec;
		vector<int> query_blade_map;
		vector<vector<float> > bladeData_vec;

		UtlGenie::loadDataQuery_mulBlades( fileHolder,  fcol_start,  fcol_end,  queryNumPerBlade,
				gq_dim_vec[gq_item_num-1],  contStep,
					 query_master_vec,
					 query_blade_map,
					 bladeData_vec
				);

		tsp.conf_TSGPUManager_blade(bladeData_vec);


		setup_contGroupQuery(tsp,
				query_master_vec, query_blade_map,  gq_dim_vec, gq_item_num,
				sc_band, windowDim);


		tsp.continous_topkQuery_dtw(topk, contStep);


//		cout<<"for test continous_topkQuery_dtw_byScanLB"<<endl;
//		//disable the enhanced lower bound
//		tsp.setEnhancedLowerBound_sel(0);
//		cout<<" setting the enhancedLowerBound_sel as:"<<tsp.getEnhancedLowerBound_sel()<<endl;
//		tsp.continous_topkQuery_dtw_byScanLB(topk, contStep);
//


}



void WrapperTSProcess::runDTW_groupQuery_onDodgers_continuousQuery(){

	//string dataHolder = "data/Dodgers/dodgers_clean";
	//int dataFileCol = 1;
	//string dataHolder = "data/isp/isp_normal";
	//int dataFileCol = 1;

	string dataHolder = "data/pems/pems_all_znormal";
	int dataFileCol = 2;

	cout << "Starting: WrapperTSProcess::runDTW_groupQuery_onDodgers_continuousQuery()"<< endl;

	int windowDim =16;
	int sc_band = 8;//windowDim/2;
	int bladeNum = 1024;

	//configuration for query
	int topk = 32; //3;
	int queryLenLoaded = 148;
	int queryLen = 128;
	int groupQueryNum = bladeNum;
	int gq_dim_vec[3] = {32,64,96};
	int gq_item_num = 3;
	int enhancedLowerBound_sel = 2;


	TSProcessor<float> tsp;
	//depressed_setupOnIntegerData_BladeAndIndex(dataHolder,fcol, tsp, numOfDocToExpand, bits_for_value, bucketWidth,  winDim, bladeNum, index_totalDimensions);
	tsp.setEnhancedLowerBound_sel(enhancedLowerBound_sel);
	cout<<" enhancedLowerBound:"<<tsp.getEnhancedLowerBound_sel()<<endl;
	setupBlade(dataHolder, dataFileCol, tsp, bladeNum);

	UtlGenie::prepareGroupQueryFile_float( dataHolder,  dataFileCol,   groupQueryNum,  queryLenLoaded);
	setupOnSeqTest_continuousQuery(dataHolder, tsp, queryLenLoaded,
			groupQueryNum, sc_band, gq_dim_vec, gq_item_num, windowDim);


	tsp.continous_topkQuery_dtw(topk, queryLenLoaded - queryLen);
	//tsp.continous_topkQuery_dtw(topk,0);

}



void WrapperTSProcess::runDTW_groupQuery_onDodgers(){
		//string dataHolder = "data/Dodgers/dodgers_clean";
		//int dataFileCol = 1;

		string dataHolder = "data/isp/isp_normal";
		int dataFileCol = 1;

		//string dataHolder = "data/test/seqTest_8k";


		cout << "Starting: WrapperTSProcess::runDTW_groupQuery_OnDodgers()"	<< endl;

		//configuration for blade and index
		int windowDim = 16;
		int sc_band = 8;//windowDim/2;
		int bladeNum = 512;

		//configuration for query
		int topk = 16;//32;//3;
		int maxQueryLen = 32;
		int groupQueryNum = bladeNum;
		int gq_dim_vec[1] = {32};
		int gq_item_num = 1;
		UtlGenie::prepareGroupQueryFile_float( dataHolder,  dataFileCol,   groupQueryNum,  gq_dim_vec[gq_item_num-1]);

		this->Genie_topkQuery(dataHolder, dataFileCol,//indicate the data file and loading which column
						 windowDim, sc_band, bladeNum, //parameter for building index: bits_for_value - how many bits to store the value in low bits; windowDim - dimension of window query; bladeNum - number of data blades
						topk,groupQueryNum, gq_dim_vec,  gq_item_num,  maxQueryLen //parameter for query:
						 );


}
//
//void WrapperTSProcess::depressed_runDTW_groupQuery_OnDodgers(){
//	cout << "Starting: WrapperTSProcess::runDTW_groupQuery_OnDodgers()"	<< endl;
//
//
//	string dataHolder = DATA_HOLDER;//"data/Dodgers/Dodgers";
//	int dataFileCol = DATA_HOLDER_COL;//1;
//
//
//	//configuration for blade and index
//	int bits_for_value = 7;
//	//float bucketWith = 1.;
//	int windowDim = WINDOW_DIMENSION;//4;
//	int bladeNum = BLADE_NUM;//2;
//	//int index_totalDimensions = bladeNum*winDim;
//
//
//	//configuration for query
//	int topk = TOP_K;//10; //3;
//	int numOfDocToExpand = 10000;
//	int sc_band =SC_BAND;// winDim / 2;
//	int maxQueryLen = QUERY_MAX_LEN;// 16;
//	int groupQueryNum = GROUPQUERY_NUM;//bladeNum;
//
//	//int gq_dim_vec[3] = { 8, 12, 16 };
//	int gq_item_num = GROUPQUERY_ITEM_NUM;//3;
//
//	int* gq_dim_vec = new int [gq_item_num];
//	for(int i=0;i<gq_item_num;i++){
//		gq_dim_vec[i] = (2+i)*windowDim;
//	}
//
//
//	int upSearchBoundExtreme = 95;
//	int downSearchBoundExtreme = 0;
//
//	depressed_Genie_topkQuery(
//			dataHolder, dataFileCol,//indicate the data file and loading which column
//			bits_for_value, windowDim, bladeNum, //parameter for building index: bits_for_value - how many bits to store the value in low bits; windowDim - dimension of window query; bladeNum - number of data blades
//			topk,groupQueryNum, gq_dim_vec, gq_item_num, maxQueryLen, //parameter for query:
//			numOfDocToExpand, sc_band, upSearchBoundExtreme, downSearchBoundExtreme//parameter for the configuration of queries
//			);
//
//	delete[] gq_dim_vec;
//
//}


void WrapperTSProcess::Genie_topkQuery(
		string dataHolder, int dataFileCol,//indicate the data file and loading which column
		int windowDim, int sc_band, int bladeNum, //parameter for building index: windowDim - dimension of window query; bladeNum - number of data blades
		int topk,int groupQueryNum,int* gq_dim_vec, int gq_item_num, int maxQueryLen //parameter for query:
	){



	TSProcessor<float> tsp;
	//( string dataFileHolder, int dataFile_col, TSProcessor<float>& tsp,  int winDim, int bladeNum)
	setupBlade(dataHolder,  dataFileCol,  tsp, bladeNum);

	setupQuery(dataHolder, tsp, maxQueryLen, groupQueryNum, windowDim, sc_band,
			 gq_dim_vec, gq_item_num);

	tsp.topkQuery_dtw(topk);


}

void WrapperTSProcess::depressed_Genie_topkQuery(
		string dataHolder, int dataFileCol,//indicate the data file and loading which column
		int bits_for_value, int windowDim, int bladeNum, //parameter for building index: bits_for_value - how many bits to store the value in low bits; windowDim - dimension of window query; bladeNum - number of data blades
		int topk,int groupQueryNum,int* gq_dim_vec, int gq_item_num, int maxQueryLen, //parameter for query:
		int numOfDocToExpand, int sc_band, int upSearchBoundExtreme, int downSearchBoundExtreme//parameter for the configuration of queries
		){

	int index_totalDimensions = bladeNum * windowDim;
	float bucketWith = 1;

	TSProcessor<int> tsp;
	depressed_setupOnIntegerData_BladeAndIndex(dataHolder, dataFileCol, tsp, numOfDocToExpand,
			bits_for_value, bucketWith, windowDim, bladeNum,
			index_totalDimensions);
	depressed_setupOnIntegerData_Query(dataHolder, tsp, maxQueryLen, groupQueryNum, sc_band,
			gq_dim_vec, gq_item_num, windowDim, upSearchBoundExtreme,
			downSearchBoundExtreme);

	tsp.depressed_topkQuery_dtw(topk);

}

