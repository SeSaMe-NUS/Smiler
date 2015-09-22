/*
 * WrapperDTW.cpp
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#include "WrapperScan.h"
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <time.h>

#include "../tools/BladeLoader.h"
#include "Scan/GPUScan.h"
#include "Scan/CPUScan.h"
#include "UtlGenie.h"

using namespace std;

#define TOPK 3
#define DIMENSIONNUM 16
#define QUERYNUM 2


WrapperScan::WrapperScan() {
	// TODO Auto-generated constructor stub

}

WrapperScan::~WrapperScan() {
	// TODO Auto-generated destructor stub
}


void loadDataFloat(const string queryFile, const string dataFile, int dataFile_col,
		vector<vector<float> >& qdata, vector<float>& data) {

	DataProcess dp;
	dp.ReadFileFloat(dataFile.c_str(), dataFile_col, data);
	//cout << "data size:" << data.size() << endl;
	//dp.printMaxMin();

	dp.ReadFileFloat(queryFile.c_str(), qdata);

	//cout << "query size:" << qdata.size() << endl;

}


void loadDataInt(const string queryFile, const string dataFile, int dataFile_col,
		vector<vector<int> >& qdata, vector<int>& data) {

	DataProcess dp;
	dp.ReadFileInt(dataFile.c_str(), dataFile_col, data);
	//cout << "data size:" << data.size() << endl;
	//dp.printMaxMin();

	dp.ReadFileInt(queryFile.c_str(), qdata);

	//cout << "query size:" << qdata.size() << endl;

}

int WrapperScan::runCPUEu() {
	int topk = TOPK;
	int dimensionNum = DIMENSIONNUM;
	int queryNum = QUERYNUM;
	string dataFileHolder = "data/Dodgers/Dodgers";
	int dataFile_col = 1;

	vector<vector<float> > qdata;
	vector<float> data;

	stringstream ssDataFile;
	ssDataFile << dataFileHolder << ".csv";
	string dataFile = ssDataFile.str();
	cout << dataFile << endl;

	stringstream ssQueryFile;
	ssQueryFile << dataFileHolder << "_d" << dimensionNum << "_q" << queryNum
			<< "_dir.query";
	string queryFile = ssQueryFile.str();
	cout << queryFile << endl;

	loadDataFloat(queryFile, dataFile, dataFile_col, qdata, data);

	CPUScan cscan;

	cscan.computTopk_float_eu(qdata, topk, data);

	return 0;

}

int WrapperScan::runGPUEu() {
	int topk = TOPK;
	int dimensionNum = DIMENSIONNUM;
	int queryNum = QUERYNUM;
	string dataFileHolder = "data/Dodgers/Dodgers";
	int dataFile_col = 1;

	vector<vector<float> > qdata;
	vector<float> data;

	stringstream ssDataFile;
	ssDataFile << dataFileHolder << ".csv";
	string dataFile = ssDataFile.str();
	cout << dataFile << endl;

	stringstream ssQueryFile;
	ssQueryFile << dataFileHolder << "_d" << dimensionNum << "_q" << queryNum
			<< "_dir.query";
	string queryFile = ssQueryFile.str();
	cout << queryFile << endl;

	loadDataFloat(queryFile, dataFile, dataFile_col, qdata, data);

	GPUScan gscan;
	gscan.computTopk_eu_float(qdata, topk, data);

	return 0;
}

int WrapperScan::runCpuDtw_scBand() {

	int topk = 8;
	int sc_band = 2;
	vector<vector<int> > qdata;
	vector<int> data;
	string dataFile = "data/Dodgers/Dodgers.csv";
	int dataFile_col = 1;
	string queryFile = "data/Dodgers/Dodgers_ql16_gqn32_group.query";

	//string dataFile = "data/test/sequenceTest.csv";
	//int dataFile_col  = 0;
	//string queryFile = "data/test/sequenceTest_ql16_gqn2_group.query";

	loadDataInt(queryFile, dataFile, dataFile_col, qdata, data);

	CPUScan cscan;
	long start = clock();
	cscan.computTopk_int_dtw_scBand(qdata, topk, data, sc_band);
	long end = clock();

	cout << "the time of top-" << topk << " in CPU version is:"
			<< (double) (end - start) / CLOCKS_PER_SEC << endl;

	return 0;
}

int WrapperScan::run_CPUvsGPU_scanExp() {


	string fileHolder = "data/Dodgers/Dodgers";
	int dataFile_col = 1;

	int dimNum = 64;
	int queryNum = 1000;

	int topk = 5;
	int sc_band = 4;

	run_CPUvsGPU_scanExp(fileHolder, dataFile_col, dimNum, queryNum, topk, sc_band);


	return 0;

}

/**
 * TODO:
 * use the first Lvec[max] as the test data to find topk, the query with larger than Lvec[max] and smaller than Lvec[max]+mulSteps are test Y label data
 */
void alignQuery_oneStep(vector<vector<float> >& query_master_vec,
		vector<vector<float> > & _query_item_vec,
		int xtst_len, int xtst_offset){
	_query_item_vec.clear();
	_query_item_vec.resize(query_master_vec.size());

	for(int i=0;i<query_master_vec.size();i++){

		assert(query_master_vec[i].size()>=xtst_len+xtst_offset);
		_query_item_vec[i].resize(xtst_len);
		std::copy(query_master_vec[i].begin()+xtst_offset,query_master_vec[i].begin()+xtst_offset+xtst_len,
				_query_item_vec[i].begin());
	}
}


void alignGroupQuery_oneStep(
	vector<vector<float> >& query_master_vec,
	vector<vector<float> > & _query_item_vec,
	int* query_item_len_vec,int query_item_num, int xtst_offset
	){

	_query_item_vec.clear();
	_query_item_vec.resize(query_master_vec.size()*query_item_num);

	for(int i=0;i<query_master_vec.size();i++){
		for(int j=0;j<query_item_num;j++){
			assert(query_master_vec[i].size()>=query_item_len_vec[j]+xtst_offset);

			_query_item_vec[i*query_item_num+j].resize(query_item_len_vec[j]);

			int query_item_offset = query_item_len_vec[query_item_num-1]-query_item_len_vec[j];//

			std::copy(query_master_vec[i].begin()+xtst_offset+query_item_offset,
					  query_master_vec[i].begin()+xtst_offset+query_item_offset+query_item_len_vec[j],
					  _query_item_vec[i*query_item_num+j].begin());

		}
	}


}
//
//void loadDataQuery_mulBlades(string fileHolder, int fcol_start, int fcol_end, int queryNumPerBlade, int queryLen, int contStep,
//		vector<vector<float> >& query_master_vec,
//		vector<int>& query_blade_map,
//		vector<vector<float> >& bladeData_vec
//	){
//
//	int bladeNum = fcol_end-fcol_start+1;
//	string dataFile = UtlGenie::getDataFile(fileHolder);
//
//	query_master_vec.reserve(bladeNum*queryNumPerBlade);
//	query_blade_map.reserve(bladeNum*queryNumPerBlade);
//	bladeData_vec.resize(bladeNum);
//
//	cout<<"=====loadDataQuery_mulBlades()::start loading data..."<<endl;
//	DataProcess dp;
//	vector<vector<float> > file_data;
//	dp.ReadFileFloat_byCol(dataFile.c_str(),file_data);
//
//
//	cout<<"=====loadDataQuery_mulBlades()::finished loading data..."<<endl;
//
//	QuerySampler qser;
//	cout<<"=====loadDataQuery_mulBlades()::start loading query..."<<endl;
//	for(int i=fcol_start;i<=fcol_end;i++){
//		vector<vector<float> > query;
//		bladeData_vec[i-fcol_start] = file_data[i];
//		//dp.ReadFileFloat(dataFile.c_str(), i, bladeData_vec[i-fcol_start]);//
//		qser.getSampleQuery_flt( bladeData_vec[i-fcol_start], queryNumPerBlade , queryLen+contStep, query);
//		for(int j=0;j<query.size();j++){
//			query_master_vec.push_back(query[j]);
//			query_blade_map.push_back(i-fcol_start);
//		}
//		//cout<<"====run_CPUvsGPU_scanExp_cont()::finished loading query of column:"<<i<<endl;
//	}
//	cout<<"=====loadDataQuery_mulBlades()::finished loading query..."<<endl;
//}

void WrapperScan::run_CPUvsGPU_scanExp_cont(string fileHolder,int fcol_start, int fcol_end,
		int queryNumPerBlade,int queryLen, int contStep,int topk, int sc_band){

	vector<vector<float> > query_master_vec;
	vector<int> query_blade_map;
	vector<vector<float> > bladeData_vec;

	UtlGenie::loadDataQuery_mulBlades( fileHolder,  fcol_start,  fcol_end,  queryNumPerBlade,  queryLen,  contStep,
			 query_master_vec,
			 query_blade_map,
			 bladeData_vec
		);

	run_CPUvsGPU_scanExp_cont(query_master_vec, bladeData_vec,
			query_blade_map, queryLen, contStep, topk, sc_band);
}


void WrapperScan::run_CPUvsGPU_scanExp_cont(vector<vector<float> >& query_master_vec, vector<vector<float> >& bladeData_vec,
		vector<int>& query_blade_map, int queryLen, int contStep, int topk, int sc_band){


	cout<<" CPU and GPU scan with continuous steps:"<<contStep<<endl;
	vector<vector<topNode> > topk_result;
	vector<int> topk_vec(query_master_vec.size(), topk);

	struct timeval tim;
	double t_start, t_end;

	GPUScan gscan;
	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout<<"run GPU Scan with time continuous step:"<<contStep<<endl;
	for(int i=0;i<contStep;i++){
		vector<vector<float> > query_item_vec;

		alignQuery_oneStep( query_master_vec,
				query_item_vec,
				queryLen, i);

		gscan.computTopk_dtw_modulus_float(query_item_vec, query_blade_map,
					bladeData_vec, topk_vec, topk_result);
	}
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of no compress vanilla GPU scan per contStep:" << (t_end - t_start)/contStep << " s"
				<< endl;


	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//==== Fast GPU scan
	cout<<"run Fast GPU scan continuous step:"<<contStep<<endl;
	for(int i=0;i<contStep;i++){
		vector<vector<float> > query_item_vec;

		alignQuery_oneStep( query_master_vec,
						query_item_vec,
						queryLen, i);

		gscan.computTopk_dtw_scBand_float(query_item_vec, query_blade_map, bladeData_vec,
					topk_vec, sc_band, topk_result);
		//gscan.printResult(topk_result);

	}
	//====
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of Fast GPU scan  per contStep:" << (t_end - t_start)/contStep << " s" << endl;


	CPUScan cscan;
	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//=== CPU scan
	cout<<"run CPU scan continuous step:"<<contStep/10<<endl;
	for (int i = 0; i < contStep/10; i++) {
		vector<vector<float> > query_item_vec;

		alignQuery_oneStep(query_master_vec, query_item_vec, queryLen, i);

		cscan.CPU_computTopk_Dtw_earlyStop(query_item_vec, query_blade_map,
				topk_vec, bladeData_vec, sc_band);//
	}

	//===
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of CPU scan per contStep:" << (t_end - t_start)/(contStep/10) << " s" << endl;
}

void WrapperScan::run_CPUvsGPU_scanExp_contGroupQuery(
		string fileHolder,int fcol_start, int fcol_end,
		int queryNumPerBlade,
		int* query_item_len_vec, int query_item_num,
		int contStep,int topk, int sc_band){

	vector<vector<float> > masterQuery_vec;
	vector<int> masterQuery_blade_map;
	vector<vector<float> > bladeData_vec;

	UtlGenie::loadDataQuery_mulBlades( fileHolder,  fcol_start,  fcol_end,  queryNumPerBlade,
			 query_item_len_vec[query_item_num-1],  contStep,
			 masterQuery_vec,
			 masterQuery_blade_map,
			 bladeData_vec
		);

	//the function for testing the time cost to compute the lower bound
//	run_GPU_dtwEnlb_contGroupQuery(
//						masterQuery_vec,
//						bladeData_vec,
//						masterQuery_blade_map,
//						query_item_len_vec, query_item_num,
//						contStep, topk, sc_band);


	run_CPUvsGPU_scanExp_contGroupQuery(
			masterQuery_vec,
			bladeData_vec,
			masterQuery_blade_map,
			query_item_len_vec, query_item_num,
			contStep, topk, sc_band);
}



void WrapperScan::run_GPU_dtwEnlb_contGroupQuery(
		vector<vector<float> >& masterQuery_vec,
		vector<vector<float> >& bladeData_vec,
		vector<int>& masterQuery_blade_map, int* query_item_len_vec,
		int query_item_num, int contStep, int topk, int sc_band) {


	vector<vector<topNode> > topk_result;
		vector<int> topk_vec(masterQuery_vec.size()*query_item_num, topk);

		//re-factor the query_blade_map
		vector<int>groupQuery_blade_map(masterQuery_blade_map.size()*query_item_num);
		for(int i=0;i<groupQuery_blade_map.size();i++){
			groupQuery_blade_map[i]=masterQuery_blade_map[i/query_item_num];
		}



	struct timeval tim;
	double t_start, t_end;
	double t_align_start = 0, t_align_end = 0, t_align = 0;
	gettimeofday(&tim, NULL);



	GPUScan gscan;
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//====test  compute lower bound ==
	cout << "run test compute lower bound for group Query with continuous step:"
			<< contStep << endl;
	for (int i = 0; i < contStep; i++) {
		vector<vector<float> > groupQuery_item_vec;

		alignGroupQuery_oneStep(masterQuery_vec, groupQuery_item_vec,
				query_item_len_vec, query_item_num, i);

		gscan.computeTopk_dtwEnlb_scBand_float(groupQuery_item_vec,
				groupQuery_blade_map, bladeData_vec, topk_vec, sc_band,
				topk_result);
		//gscan.printResult(topk_result);

	}
	//====test  compute lower bound ==
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of Fast GPU scan  per contStep:"
			<< (t_end - t_start) / contStep << " s" << endl;

}


void WrapperScan::run_CPUvsGPU_scanExp_contGroupQuery(
		vector<vector<float> >& masterQuery_vec,
		vector<vector<float> >& bladeData_vec,
		vector<int>& masterQuery_blade_map,
		int* query_item_len_vec, int query_item_num,
		int contStep, int topk, int sc_band){



	cout<<" CPU and GPU scan for group Query with continuous steps:"<<contStep<<endl;
	vector<vector<topNode> > topk_result;
	vector<int> topk_vec(masterQuery_vec.size()*query_item_num, topk);

	//re-factor the query_blade_map
	vector<int>groupQuery_blade_map(masterQuery_blade_map.size()*query_item_num);
	for(int i=0;i<groupQuery_blade_map.size();i++){
		groupQuery_blade_map[i]=masterQuery_blade_map[i/query_item_num];
	}

	struct timeval tim;
	double t_start, t_end;
	double t_align_start=0,t_align_end=0, t_align=0;


	GPUScan gscan;

	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout<<"run GPU Scan for group Query with time continuous step:"<<contStep<<endl;
	for(int i=0;i<contStep;i++){
		vector<vector<float> > groupQuery_item_vec;
		gettimeofday(&tim, NULL);
		t_align_start=tim.tv_sec + (tim.tv_usec / 1000000.0);;
		alignGroupQuery_oneStep(
			masterQuery_vec,
			groupQuery_item_vec,
			query_item_len_vec,query_item_num, i
			);
		gettimeofday(&tim, NULL);
		t_align_end=tim.tv_sec + (tim.tv_usec / 1000000.0);;
		t_align+=t_align_end-t_align_start;

		gscan.computTopk_dtw_modulus_float(groupQuery_item_vec, groupQuery_blade_map,
					bladeData_vec, topk_vec, topk_result);
	}
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of no compress vanilla GPU scan per contStep:" << (t_end - t_start)/contStep << " s"
				<< endl;
	cout<<" time for align query per ContStep is:"<<t_align/contStep<<" s"<<endl;




	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//==== Fast GPU scan
	cout<<"run Fast GPU scan  for group Query with continuous step:"<<contStep<<endl;
	for(int i=0;i<contStep;i++){
		vector<vector<float> > groupQuery_item_vec;

		alignGroupQuery_oneStep(
					masterQuery_vec,
					groupQuery_item_vec,
					query_item_len_vec,query_item_num, i
		);

		gscan.computTopk_dtw_scBand_float(groupQuery_item_vec, groupQuery_blade_map, bladeData_vec,
					topk_vec, sc_band, topk_result);
		gscan.printResult(topk_result);//with debug purpose//

	}
	//====
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of Fast GPU scan  per contStep:" << (t_end - t_start)/contStep << " s" << endl;



	//
	CPUScan cscan;
	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//=== CPU scan
	cout<<"run CPU scan for group Query continuous step:"<<contStep<<endl;
	for (int i = 0; i < contStep; i++) {
		vector<vector<float> > groupQuery_item_vec;

		alignGroupQuery_oneStep(
							masterQuery_vec,
							groupQuery_item_vec,
							query_item_len_vec,query_item_num, i
				);

		cscan.CPU_computTopk_Dtw_earlyStop(groupQuery_item_vec, groupQuery_blade_map,
				topk_vec, bladeData_vec, sc_band);//

	}


	//===
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of CPU scan per contStep:" << (t_end - t_start)/(contStep/10) << " s" << endl;
}

void WrapperScan::run_CPUvsGPU_scanExp(vector<vector<float> >& query_vec, vector<vector<float> >& bladeData_vec,
		vector<int>& query_blade_map, int topk, int sc_band){

	vector<vector<topNode> > topk_result;
	vector<int> topk_vec(query_vec.size(), topk);

	struct timeval tim;
	double t_start, t_end;

	GPUScan gscan;

	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//=========== vanilla GPU scan
	gscan.computTopk_dtw_modulus_float(query_vec, query_blade_map,
			bladeData_vec, topk_vec, topk_result);
	//==========
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of no compress vanilla GPU scan:" << t_end - t_start << " s"
			<< endl;

	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//==== Fast GPU scan
	gscan.computTopk_dtw_scBand_float(query_vec, query_blade_map, bladeData_vec,
			topk_vec, sc_band, topk_result);
	//====
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of Fast GPU scan:" << t_end - t_start << " s" << endl;

	CPUScan cscan;
	gettimeofday(&tim, NULL);
	t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);
	//=== CPU scan
	cscan.CPU_computTopk_Dtw_earlyStop(query_vec, query_blade_map, topk_vec,
			bladeData_vec, sc_band);
	//===
	gettimeofday(&tim, NULL);
	t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "time of CPU scan:" << t_end - t_start << " s" << endl;


}

void WrapperScan::run_CPUvsGPU_scanExp(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk, int sc_band) {


	string dataFile = UtlGenie::getDataFile(fileHolder);
	string queryFile = UtlGenie::getQueryFile(fileHolder, dimensionNum, queryNum);
	UtlGenie::prepareQueryFile_float( dataFile, dataFileCol,  queryFile, queryNum, dimensionNum);
	cout<<"query file: "<<queryFile<<endl;

	vector<vector<float> > query_vec;
	vector<float> data;
	loadDataFloat(queryFile, dataFile, dataFileCol, query_vec, data);

	vector<int> query_blade_map(query_vec.size(), 0);
	vector<vector<float> > bladeData_vec(query_vec.size());

	for (int i = 0; i < query_vec.size(); i++) {
		bladeData_vec[i] = data;
		query_blade_map[i] = i;
	}

	run_CPUvsGPU_scanExp( query_vec,  bladeData_vec,  query_blade_map,  topk,  sc_band);

}

int WrapperScan::depressed_runCPUvsGPU_scan_singleBlade() {

	int topk = 5;
	int sc_band = 256;
	vector<vector<float> > qdata;
	vector<float> data;
	string dataFile = "data/Dodgers/Dodgers.csv";
	int dataFile_col = 1;
	string queryFile = "data/Dodgers/Dodgers_d256_q16000_dir.query";
	loadDataFloat(queryFile, dataFile, dataFile_col, qdata, data);

	GPUScan gscan;
	//gscan.computTopk_int_dtw_scBand(qdata,topk,data,sc_band);
	gscan.computTopk_eu_float(qdata, topk, data);

	CPUScan cscan;
	//cscan.computTopk_int_dtw_scBand_earlyStop(qdata, topk, data, sc_band);

	return 0;
}

/**
 *

 void WrapperScan::DTWQuery(vector<int>& data, vector<vector<int> >& qdata, int dim, int topk,
 vector<vector<int> >& _resIdx, vector<vector<int> >& _dist){


 //copy time series into GPU
 int ts_len = data.size();//
 int* ts= (data).data();//get raw data

 int tq_num = qdata.size();
 int  res_buff_len = tq_num*(ts_len-dim+1);
 int* res_buff = new int[res_buff_len];

 //int** resIdx = new int*[tq_num];//the return result
 _resIdx.resize(tq_num);

 //prepare the query file
 int tq_len = 0;
 tq_len = dim*tq_num;
 int* tq =vec2Ddata(qdata);//convert query data into one dimesnion array


 startGPU(ts,  ts_len,  tq,  tq_num,  dim, res_buff);//use GPU to compute DTW
 //selectQueryRes(int* res_buff, int topk,int tq_num, int ts_len, int dim, vector<vector<int> >&  resIdx)
 selectQueryRes(res_buff, topk, tq_num,  ts_len,  dim, _resIdx, _dist);//select top-k query

 delete[] res_buff;
 //delete ts;
 delete[] tq;

 //return resIdx;
 }


 int WrapperScan::runDTWQueryInt(string inputFilename, string queryFile,int columnAv, int dim, int tq_num, int topk){

 DataProcess dp;
 //load data
 vector<int> data;
 dp.ReadFileInt(inputFilename.c_str(),columnAv,data);
 cout<<"load data item:"<<data.size()<<endl;
 //load query
 vector<vector<int> > qdata;
 dp.ReadFileInt(queryFile.c_str(),qdata);
 cout<<"load query items:"<<qdata.size()<<endl;

 long time=0;
 long start = clock();
 vector<vector<int> > resIdx;
 vector<vector<int> > dist;
 DTWQuery(data,  qdata,  dim,  topk, resIdx, dist);
 long end=clock();
 time = end-start;
 cout<<"the running time of GPU scan is:"<<(double)time /CLOCKS_PER_SEC <<endl;

 return 0;

 }

 int WrapperScan::runDTWQueryByInput(){


 //load time series data
 string inputFilename;
 cout << "Please enter input filename" << endl;
 cin >> inputFilename;

 cout << "Which column is going to be used? (Start with column 0)" << endl;
 int columnAv;
 cin >> columnAv;

 string queryFile;
 cout<< "Please enter query filename(note: if selecting data from datafile, please input *";
 cin >> queryFile;

 //define the query
 cout<<"Please input query number"<<endl;
 int tq_num=1;
 cin>>tq_num;

 int dim=32;
 cout<<"Please input dimensions:"<<endl;
 cin>>dim;

 int k = 8;
 cout<<"please input the top-k (NN):"<<endl;
 cin>>k;

 runDTWQueryInt(inputFilename, "", columnAv,  dim,  tq_num, k);

 return 0;
 }




 int WrapperScan::runGPUDTW(){
 //runQueryByInput();
 //runDTWQueryInt( "data/calit2/CalIt2_7.csv", "data/calit2/CalIt2_7_d8_q16_dir.query", 3,  8,  16,  3);
 runDTWQueryInt( "data/test/sequenceTest.csv", "data/test/sequenceTest_ql16_gqn2_group.query", 0,  16,  2,  10);
 return 0;
 }
 *
 */

