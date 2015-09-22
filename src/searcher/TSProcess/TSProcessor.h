/*
 * TSPROCESSOR.h
 *
 *  Created on: May 8, 2014
 *      Author: zhoujingbo
 */

#ifndef TSPROCESSOR_H_
#define TSPROCESSOR_H_
#include "../GPUKNN/UtlGPU.h"
#include <assert.h>
#include "UtlTSProcess.h"
#include "TSGPUManager.h"
#include <cuda_profiler_api.h>

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sys/time.h>
//using namespace std;


template <class T>
class TSProcessor {
public:

	TSProcessor():tsGpuManager(){
		winDim = 0;
		groupQuery_maxLen=0;
	}

	virtual ~TSProcessor(){
	}

public:

	int getGroupQueryNum(){
		return tsGpuManager.getGroupQueryNum();
	}


	/**
	 * TODO:
	 * make the configuration of the idnex in GPU
	 *index_totalDimensions: the total dimensions for all blades in the GPU, suppose there are 16 blades and each blades with 8 window dimensions, the index_totalDimensions = 16*8;
	 */
	void depressed_conf_TSGPUManager_index_blade(const int numOfDocToExpand, const int winDim,  string idxPath, int bits_for_value, float bucketWidth,
			const int bladeNum, vector<float>& blade_data_vec, vector<int>& blade_len_vec) {

		GPUSpecification test_spec;
		test_spec.numOfDocToExpand = numOfDocToExpand;
		int index_totalDimensions = winDim * bladeNum;
		test_spec.totalDimension = index_totalDimensions;
		test_spec.default_disfuncType = 2;
		test_spec.default_upwardDistBound = 0;
		test_spec.default_downwardDistBound = 0;

		test_spec.invertedListPath = idxPath;

		test_spec.indexDimensionEntry.resize(test_spec.totalDimension);

		for (int i = 0; i < test_spec.totalDimension; i++) {

			//test_spec.indexDimensionEntry[i].minDomain = 0;           // dangerous:!!
			//test_spec.indexDimensionEntry[i].maxDomain = pow(2,bits_for_value) -1;		 // dangerous:!!
			test_spec.indexDimensionEntry[i].bucketWidth = bucketWidth;


		}


		tsGpuManager.conf_dev_blade(bladeNum,blade_data_vec,blade_len_vec);
		tsGpuManager.depressed_conf_dev_index(test_spec,winDim);
		this->winDim = winDim;
	}

	void conf_TSGPUManager_blade(const int bladeNum, vector<float>& blade_data_vec, vector<int>& blade_len_vec){
		tsGpuManager.conf_dev_blade(bladeNum,blade_data_vec,blade_len_vec);

	}

	void conf_TSGPUManager_blade(vector<vector<float> >& bladeData_vec){
		tsGpuManager.conf_dev_blade(bladeData_vec);
	}

	/**
	 * TODO:
	 * setup the query
	 *
	 * upSearchExtreme_vec: record the maximum value for each group query
	 * downSearchExtreme_vec:record the minimum value for each group query
	 *
	 * groupQuery_info_set: record all information for groupQuery, i.e. item number, item query diemnsions.
	 *
	 * NOTE: in this function, we do not set queryData, which is used for continous prediction
	 */
	void depressed_conf_TSGPUManager_query(int sc_band, int queryLen,int itemNum_perGroup,
			vector<GroupQuery_info*>& groupQuery_info_set,
			vector<int>& upSearchExtreme_vec, vector<int>& downSearchExtreme_vec){

		this-> groupQuery_maxLen = queryLen;
		int winNumPerGroup = queryLen - winDim + 1;

		tsGpuManager.depressed_conf_dev_query(groupQuery_info_set,upSearchExtreme_vec, downSearchExtreme_vec,
				winNumPerGroup, itemNum_perGroup, sc_band);

	}

	void conf_TSGPUManager_query(int winDim,int sc_band, int queryLen, int itemNum_perGroup, vector<GroupQuery_info*>& groupQuery_info_set){
		this->winDim = winDim;
		this-> groupQuery_maxLen = queryLen;
		int winNumPerGroup = queryLen - winDim + 1;

		tsGpuManager.conf_dev_query(groupQuery_info_set,
				winNumPerGroup, itemNum_perGroup,winDim,sc_band);
	}

	void depressed_conf_TSGPUManager_query_LKKeogh_ACDUpdate(){
		tsGpuManager.enable_winQuery_ACDUpdate_forLBKeogh();//
		tsGpuManager.depressed_init_winQuery_ACDUpdate_forLBKeogh();
	}

	void depressed_conf_TSGPUManager_continuousQuery(int sc_band, int queryLen,int itemNum_perGroup,
			vector<GroupQuery_info*>& groupQuery_info_set,
			vector<vector<T> >& continuous_queryData,
			vector<int>& upSearchExtreme_vec, vector<int>& downSearchExtreme_vec){

		depressed_conf_TSGPUManager_query(sc_band,  queryLen, itemNum_perGroup,
					 groupQuery_info_set,
					 upSearchExtreme_vec, downSearchExtreme_vec);
		conf_TSGPUManager_continuousQueryData(continuous_queryData,queryLen);
		depressed_conf_TSGPUManager_query_LKKeogh_ACDUpdate();

	}

	void conf_TSGPUManager_continuousQuery(int windowDim, int sc_band, int queryLen,int itemNum_perGroup,
			vector<GroupQuery_info*>& groupQuery_info_set,
			vector<vector<T> >& continuous_queryData
			){

		conf_TSGPUManager_query(windowDim,sc_band,  queryLen, itemNum_perGroup, groupQuery_info_set);
		conf_TSGPUManager_continuousQueryData(continuous_queryData,queryLen);
		//depressed_conf_TSGPUManager_query_LKKeogh_ACDUpdate();

	}

	void conf_TSGPUManager_continuousQueryData(vector<vector<T> >& continuous_queryData, int endIndicator){

		this->continuous_queryData_endIndicator.resize(continuous_queryData.size(), endIndicator);
		this->continuous_queryData.resize(continuous_queryData.size());
		//std::copy(continuous_queryData.begin(),continuous_queryData.end(),this->continuous_queryData.begin());

		for(int i=0;i<continuous_queryData.size();i++){
			this->continuous_queryData[i].resize(continuous_queryData[i].size());
			//std::copy(continuous_queryData[i].begin(),continuous_queryData[i].end(),this->continuous_queryData[i].begin());
			for(int j=0;j<continuous_queryData[i].size();j++){
				this->continuous_queryData[i][j] = (float)continuous_queryData[i][j];
			}
		}


	}

	void topkQuery_dtw(int topk){

		cout<<"starting top-k query with topk ="<<topk<<endl;
		tsGpuManager.exact_topkQuery_DTW(topk);//
		cout<<"end top-k query"<<endl;

		tsGpuManager.print_d_groupQuerySet_topkResults(topk);
		//tsGpuManager.print_proflingTime();
		host_vector<CandidateEntry> topkRes;
		tsGpuManager.getTopkResults(topkRes);

	}



	void depressed_topkQuery_dtw(int topk){

		cout<<"starting top-k query with topk ="<<topk<<endl;
		//tsGpuManager.exact_topk_query_DTW_randomSelect_bucketUnit(topk);//.exact_TopK_query_DTW_bucketUnit(topk);
		tsGpuManager.depressed_exact_TopK_query_DTW_fastSelect_bucketUnit(topk);

		cout<<"end top-k query"<<endl;


		tsGpuManager.print_d_groupQuerySet_topkResults(topk);
		tsGpuManager.print_proflingTime();
		host_vector<CandidateEntry> topkRes;
		tsGpuManager.getTopkResults(topkRes);

	}

	/**
	 * stepDelta: predict for next stepDelta step
	 * TODO:
	 * update window query, appending new sliding windows to gpuManager
	 * update data array of GroupQueryInfo
	 */
	void depressed_conf_contQuery_nextStep(int stepDelta){

		vector<vector<float> > gqi_data_set;
		vector<int> gqi_newData_indicator;//after (and inclusive) this position, every elements in qi_data_set are new data
		gqi_data_set.resize(continuous_queryData.size());
		gqi_newData_indicator.resize(continuous_queryData.size());

		for(int i=0;i<continuous_queryData.size();i++){
			continuous_queryData_endIndicator[i] += stepDelta;
			int gq_indicator_start = continuous_queryData_endIndicator[i] - groupQuery_maxLen ;
			gqi_newData_indicator[i] = groupQuery_maxLen-stepDelta;//
			gqi_data_set[i].resize(groupQuery_maxLen,0);

			//new query data array
			std::copy(continuous_queryData[i].begin()+gq_indicator_start,
					continuous_queryData[i].begin()+gq_indicator_start+groupQuery_maxLen,
					gqi_data_set[i].begin());

		}

		tsGpuManager.depressed_update_ContQueryInfo_set(gqi_data_set,gqi_newData_indicator);
		tsGpuManager.depressed_reset_TSGPUMananger_forGroupQuery();

	}


	/**
	 * stepDelta: predict for next stepDelta step
	 * TODO:
	 * update window query, appending new sliding windows to gpuManager
	 * update data array of GroupQueryInfo
	 */
	void conf_contQuery_nextStep(int stepDelta){

		vector<vector<float> > gqi_data_set;
		vector<int> gqi_newData_indicator;//after (and inclusive) this position, every elements in qi_data_set are new data
		gqi_data_set.resize(continuous_queryData.size());
		gqi_newData_indicator.resize(continuous_queryData.size());

		tsGpuManager.profile_clockTime_start("conf_contQuery_nextStep():forloop");
		for(int i=0;i<continuous_queryData.size();i++){
			continuous_queryData_endIndicator[i] += stepDelta;
			int gq_indicator_start = continuous_queryData_endIndicator[i] - groupQuery_maxLen ;
			gqi_newData_indicator[i] = groupQuery_maxLen-stepDelta;//
			gqi_data_set[i].resize(groupQuery_maxLen,0);

			//new query data array
			std::copy(continuous_queryData[i].begin()+gq_indicator_start,
					continuous_queryData[i].begin()+gq_indicator_start+groupQuery_maxLen,
					gqi_data_set[i].begin());

		}
		tsGpuManager.profile_clockTime_end("conf_contQuery_nextStep():forloop");

		tsGpuManager.profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set()");
		tsGpuManager.update_ContQueryInfo_set(gqi_data_set,gqi_newData_indicator);
		tsGpuManager.profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set()");

		tsGpuManager.profile_clockTime_start("conf_contQuery_nextStep():reset_TSGPUMananger_forGroupQuery()");
		tsGpuManager.reset_TSGPUMananger_forGroupQuery();
		tsGpuManager.profile_clockTime_end("conf_contQuery_nextStep():reset_TSGPUMananger_forGroupQuery()");

	}


	void depressed_continous_topkQuery_dtw(int topk, int conSteps){

		cout<<"do depressed_continuous prediction step 0"<<endl;
		depressed_topkQuery_dtw(topk);//the first step

		for(int i=1;i<conSteps;i++){
			cout<<endl<<endl;
			cout<<"*********************do depressed_continuous prediction step "<<i<<endl;

			depressed_conf_contQuery_nextStep(1);

			depressed_topkQuery_dtw(topk);//the next step
		}

	}


	//the purpose for testing
	void continous_topkQuery_dtw_byScanLB(int topk, int conSteps){
		tsGpuManager.exact_topkQuery_DTW_contFirstStep(topk);//

		tsGpuManager.profile_clockTime_start("time continous_topkQuery_dtw_byScanLB()");

		for(int i=1;i<conSteps;i++){
			tsGpuManager.profile_clockTime_start("conf_contQuery_nextStep()");
			conf_contQuery_nextStep(1);
			tsGpuManager.profile_clockTime_end("conf_contQuery_nextStep()");

			tsGpuManager.profile_clockTime_start("A sum:exact_topkQuery_DTW_contNextStep()");
			tsGpuManager.profile_cudaTime_start("A sum:exact_topkQuery_DTW_contNextStep()");
			tsGpuManager.exact_topkQuery_DTW_contNextStep(topk);
			tsGpuManager.profile_cudaTime_end("A sum:exact_topkQuery_DTW_contNextStep()");
			tsGpuManager.profile_clockTime_end("A sum:exact_topkQuery_DTW_contNextStep()");
		}

		tsGpuManager.profile_clockTime_end("time continous_topkQuery_dtw_byScanLB()");

		tsGpuManager.print_proflingTime_perContStep();
		tsGpuManager.print_proflingTime( );

	}

	void contFirstTopk(int topk,host_vector<CandidateEntry>& topResults, host_vector<int>& topResults_size){

		tsGpuManager.exact_topkQuery_DTW_contFirstStep(topk);

		topResults = tsGpuManager.d_groupQuerySet_topkResults;
		topResults_size = tsGpuManager.d_groupQuerySet_topkResults_size;
	}

	void contNextTopk(int topk,host_vector<CandidateEntry>& topResults, host_vector<int>& topResults_size, int stepDelta=1){

		conf_contQuery_nextStep(stepDelta);
		tsGpuManager.exact_topkQuery_DTW_contNextStep(topk);


		topResults = tsGpuManager.d_groupQuerySet_topkResults;
		topResults_size = tsGpuManager.d_groupQuerySet_topkResults_size;

	}

	void continous_topkQuery_dtw(int topk, int conSteps){

		//cout<<"do continuous prediction step 0"<<endl;
		//topkQuery_dtw(topk);//the first step

		tsGpuManager.exact_topkQuery_DTW_contFirstStep(topk);//
		//tsGpuManager.print_d_groupQuerySet_topkResults(topk);//with debug purpose//


		tsGpuManager.profile_clockTime_start("time continous_topkQuery_dtw_allStep()");
		for(int i=1;i<conSteps;i++){
			tsGpuManager.profile_clockTime_start("time continous_topkQuery_dtw_perStep()");
			//cout<<endl<<endl;
			//cout<<"*********************do continuous prediction step "<<i<<endl;
			tsGpuManager.profile_clockTime_start("conf_contQuery_nextStep()");
			conf_contQuery_nextStep(1);
			tsGpuManager.profile_clockTime_end("conf_contQuery_nextStep()");

			tsGpuManager.profile_clockTime_start("A sum:exact_topkQuery_DTW_contNextStep()");
			tsGpuManager.profile_cudaTime_start("A sum:exact_topkQuery_DTW_contNextStep()");
			tsGpuManager.exact_topkQuery_DTW_contNextStep(topk);
			tsGpuManager.profile_cudaTime_end("A sum:exact_topkQuery_DTW_contNextStep()");
			tsGpuManager.profile_clockTime_end("A sum:exact_topkQuery_DTW_contNextStep()");

			tsGpuManager.profile_clockTime_end("time continous_topkQuery_dtw_perStep()");

			tsGpuManager.print_d_groupQuerySet_topkResults(topk);//with debug purpose
		}
		tsGpuManager.profile_clockTime_end("time continous_topkQuery_dtw_allStep()");

		//tsGpuManager.print_d_groupQuerySet_topkResults(topk);

		//tsGpuManager.print_proflingTime();
		tsGpuManager.print_proflingTime_perContStep();
		//tsGpuManager.print_proflingTime();
	}


	void getTopkQueryResult(host_vector<CandidateEntry>& topkRes){
		tsGpuManager.getTopkResults(topkRes);
	}

	/**
	 * 	enhancedLowerBound_sel:
 *						0: use d2q
 *						1: use q2d
 *						2: use max(d2q,q2d)
	 */
	void setEnhancedLowerBound_sel(int used_mode){
		tsGpuManager.enhancedLowerBound_sel=used_mode;
		//if(used_mode>0){
			tsGpuManager.enhancedLowerBound=true;
		//}else{
		//	tsGpuManager.enhancedLowerBound=false;
		//}
	}
	int getEnhancedLowerBound_sel(){
		return tsGpuManager.enhancedLowerBound_sel;
	}

	void setEnableSumUnfilteredCandidates(bool mode){
		tsGpuManager.enable_sum_unfiltered_candidates=mode;
	}

private:
	TSGPUManager tsGpuManager;
	int winDim;
	//all the group of query have the same number of sliding window query, all sliding windows have the same length,
	//-> this condition means all the group query should have the same number of maximum query length
	int groupQuery_maxLen;

	vector<vector<float> > continuous_queryData;//store the query data for multiple group query with the largest dimensions, this used for continuous prediction
	vector<int> continuous_queryData_endIndicator;//record the end position for query stored in queryData (with open inverval, exclusive), i.e. recorded next position
};


#endif /* TSPROCESSOR_H_ */
