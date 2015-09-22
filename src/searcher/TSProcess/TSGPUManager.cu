/*
 * TSGPUManager.cpp
 *
 *  Created on: May 13, 2014
 *      Author: zhoujingbo
 */

#include "TSGPUManager.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <float.h>


//add
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>



//#include "generalization.h"

#include "../GPUKNN/GPUManager.h"
#include "../GPUKNN/generalization.h"
#include "../lib/tailored_bucket_topk/bucket_topk.h"//
#include "../DataEngine/UtlDataEngine.h"
using namespace std;
using namespace thrust;


#include <iostream>
#include <fstream>
#include <sstream>

#include <cuda_profiler_api.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>


#define BILLION 1E9
#define prefix_timespec  "clock time:"
#define prefix_cudatime  "cuda time:"




void initQueryItem_LBKeogh(int sc_band,
		float* queryData,
		int queryData_start, int queryData_len, int numOfDimensionToSearch,
		float* keywords,
		vector<float>& upwardDistanceBound, vector<float>& downwardDistanceBound) {

	for (int i = 0; i < numOfDimensionToSearch; i++) {
		int j = (queryData_start + i - sc_band <= 0) ?
				0 : (queryData_start + i - sc_band);
		int s = (queryData_start + i + sc_band >= (queryData_len - 1)) ?
				(queryData_len - 1) : (queryData_start + i + sc_band);

		float up = -(float) INT_MAX; //find maximum value within [i-r,i+r]
		float down = (float) INT_MAX; //find minimum value within [i-r,i+r]

		for (; j <= s; j++) {
			if (up < queryData[j]) {
				up = queryData[j];
			}
			if (down > queryData[j]) {
				down = queryData[j];
			}
		}

		up = up - keywords[i];
		down = keywords[i] - down;

		assert(up >= 0 && down >= 0);
		upwardDistanceBound[i] = up;
		downwardDistanceBound[i] = down;

	}
}



void initQueryItem_LBKeogh(int sc_band, float* queryData, int queryData_start,
		int queryData_len, GpuWindowQuery& _query) {

	initQueryItem_LBKeogh( sc_band,
			queryData,
			queryData_start, queryData_len, _query.numOfDimensionToSearch,
			 _query.keywords.data(),
			 _query.upwardDistanceBound, _query.downwardDistanceBound);

}


/*
TSGPUManager::TSGPUManager(GPUSpecification& gpuSpec):gpuManager(gpuSpec) {
	// TODO Auto-generated constructor stub

	initMemeberVariables();

	int winMaxFid = getMaxFeatureID() ;
	init_dev_winIndex_constantMem(this->winDim, this->getMaxFeatureID());
}*/

TSGPUManager::TSGPUManager():depressed_gpuManager(){
	init_parameters_default();
}

TSGPUManager::~TSGPUManager() {
	// TODO Auto-generated destructor stub

	for(int i=0;i<h_groupQuery_info_set.size();i++){
		delete h_groupQuery_info_set[i];
	}
	h_groupQuery_info_set.clear();

	this->free_GroupQueryInfo_device(d_groupQuery_info_set);

}

void TSGPUManager::init_parameters_default(){
	this->windowNum_perGroup = 0;
	this->sc_band = 1;
	this-> winDim = 0;
	this->groupQuery_maxdimension = 0;
	this->groupQuery_item_num = 0;
	this->ts_blade_num = 0;
	enhancedLowerBound = false;
	enhancedLowerBound_sel=0;
	enable_sum_unfiltered_candidates=false;
	this->sum_unfiltered_candidates=0;
	this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = false;
	depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled = false;


}

void TSGPUManager::conf_dev_blade(const int bladeNum, const vector<float>& blade_data_vec,const vector<int>& blade_len_vec ){

	this->ts_blade_num = bladeNum;

	this->d_ts_data.resize(blade_data_vec.size());
	if(enhancedLowerBound){
		this->d_ts_data_u.resize(blade_data_vec.size());
		this->d_ts_data_l.resize(blade_data_vec.size());
	}

	thrust::copy(blade_data_vec.begin(),blade_data_vec.end(),this->d_ts_data.begin());

	this->d_ts_data_blade_endIdx.resize(blade_len_vec.size());
	thrust::copy(blade_len_vec.begin(),blade_len_vec.end(),this->d_ts_data_blade_endIdx.begin());
	thrust::inclusive_scan(d_ts_data_blade_endIdx.begin(),d_ts_data_blade_endIdx.end(),d_ts_data_blade_endIdx.begin());

	this->h_ts_blade_len.resize(blade_len_vec.size());
	thrust::copy(blade_len_vec.begin(),blade_len_vec.end(),this->h_ts_blade_len.begin());

	// compute this->d_ts_data_u.resize(blade_data_vec.size()); this->d_ts_data_l.resize(blade_data_vec.size());
	//postponed them to conf_dev_query().

}


void TSGPUManager:: conf_dev_blade(vector<vector<float> >& bladeData_vec){

	int lenSum = 0;
	vector<int> blade_len_vec(bladeData_vec.size());
	for(int i=0;i<bladeData_vec.size();i++){
		blade_len_vec[i] = bladeData_vec[i].size();
		lenSum+=blade_len_vec[i];
	}

	vector<float> bladeData;
	bladeData.reserve(lenSum);
	for(int i=0;i<bladeData_vec.size();i++){
		for(int j=0;j<bladeData_vec[i].size();j++){
			bladeData.push_back(bladeData_vec[i][j]);
		}
	}

	this->conf_dev_blade(bladeData_vec.size(), bladeData,blade_len_vec);
}


void TSGPUManager::depressed_conf_dev_index(GPUSpecification& windowQuery_gpuSpec, const int winDim){

		this->winDim = winDim;

		depressed_gpuManager.init_dimensionNum_perQuery(winDim);
		depressed_gpuManager.conf_GPUManager_GPUSpecification(windowQuery_gpuSpec);

		init_dev_windowQuery_constantMem(winDim, this->depressed_getMaxWindowFeatureNumber());


}



/**
 * TODO:
 * for all the group query items, configuration the windows queries
 * this is an auxiliary function of conf_TSGPUManager_query(),
 *
 * upSearchExtreme_vec: record the maximum value for each group query
 * downSearchExtreme_vec:record the minimum value for each group query
 *
 * 	//note: must configure h_groupQuery_info_set firstly
 *
 */
void TSGPUManager::depressed_set_windowQueryInfo_set(int sc_band, vector<GpuWindowQuery>& _windowQuerySet) {


	int winNumPerGroup = groupQuery_maxdimension - winDim + 1;
	for (int i = 0; i < this->h_groupQuery_info_set.size(); i++) {

		for (int j = 0; j < winNumPerGroup; j++) {
			GpuWindowQuery gq_sw(i * winNumPerGroup + j, h_groupQuery_info_set[i]->blade_id, winDim);

			depressed_set_windowQueryInfo_GpuQuery_newEntry(i, j, gq_sw);

			_windowQuerySet.push_back(gq_sw);
		}
	}
}



/**
 * TODO:
 * for all the group query items, configuration the windows queries
 * this is an auxiliary function of conf_TSGPUManager_query(),
 *
 *
 * 	//note: must configure h_groupQuery_info_set firstly
 *
 */
void TSGPUManager::set_windowQueryInfo_set(int sc_band, vector<GpuWindowQuery>& _windowQuerySet) {


	int winNumPerGroup = groupQuery_maxdimension - winDim + 1;
	_windowQuerySet.reserve(winNumPerGroup*h_groupQuery_info_set.size());
	this->h_windowQuery_LBKeogh_updatedLabel.resize(winNumPerGroup*h_groupQuery_info_set.size(),true);

	for (int i = 0; i < this->h_groupQuery_info_set.size(); i++) {

		for (int j = 0; j < winNumPerGroup; j++) {
			GpuWindowQuery gq_sw(i * winNumPerGroup + j, h_groupQuery_info_set[i]->blade_id, winDim);

			set_windowQueryInfo_GpuQuery_newEntry(i, j, gq_sw);

			_windowQuerySet.push_back(gq_sw);
		}
	}
}

/**
 * configuration one window query item
 *
 * slidingWin_dataIdx: the starting index (position) of a sliding window in the data array of GroupQueryInfo
 *
 */
void TSGPUManager::depressed_set_windowQueryInfo_GpuQuery_newEntry(int groupQueryId, int slidingWin_dataIdx,
		GpuWindowQuery& _gq_sw) {

	GroupQuery_info* groupQuery_info = this->h_groupQuery_info_set[groupQueryId];
	//GpuQuery gq_sw(groupQueryId*winNumPerGroup+windowQueryId, winDim);
	for (int kj = 0; kj < winDim; kj++) {
		_gq_sw.keywords[kj] = groupQuery_info->data[slidingWin_dataIdx + kj];
		_gq_sw.depressed_dimensionSet[kj].dimension = groupQueryId * winDim + kj;
		_gq_sw.depressed_upwardSearchBound[kj] = this->depressed_groupQuery_upSearchExtreme[groupQueryId] - _gq_sw.keywords[kj];
		_gq_sw.depressed_downwardSearchBound[kj] = _gq_sw.keywords[kj]
				- this->depressed_groupQuery_downSearchExtreme[groupQueryId];
	}

	initQueryItem_LBKeogh(sc_band, groupQuery_info->data, slidingWin_dataIdx,
			groupQuery_maxdimension, _gq_sw);

}



/**
 * configurate one window query item,this is to generate new window query when new data are coming
 *
 * slidingWin_dataIdx: the starting index (position) of a sliding window in the data array of GroupQueryInfo (note this is not logical window id)
 *
 */
void TSGPUManager::set_windowQueryInfo_GpuQuery_newEntry(int groupQueryId, int slidingWin_dataIdx,
		GpuWindowQuery& _gq_sw) {

	GroupQuery_info* groupQuery_info = this->h_groupQuery_info_set[groupQueryId];
	//GpuQuery gq_sw(groupQueryId*winNumPerGroup+windowQueryId, winDim);
	for (int kj = 0; kj < winDim; kj++) {
		_gq_sw.keywords[kj] = groupQuery_info->data[slidingWin_dataIdx + kj];
	}

	initQueryItem_LBKeogh(sc_band, groupQuery_info->data, slidingWin_dataIdx,
			groupQuery_maxdimension, _gq_sw);//


}

/**
 * TODO:
 * initialize and set query
 *
**
 * important assumption:
 * 1. all data in different groups have the same number of maxWinFeatureID, or their maxWinFeaturesIDs are similar and we select the maximum one.
 * 2. all the group of query have the same number of sliding window query, all sliding windows have the same length,
 * 		this means all group queries should have the same maximum length
 * 3. the minimum query dimensions of group query should be 2 time of the width of sliding windows
 *    create the sliding windows in reverse order (from tailed to head) for group query.
 * 4. all the group queries have the same number of query items
 *
 */
void TSGPUManager::depressed_conf_dev_query(vector<GroupQuery_info*>& groupQuery_info_set,
		vector<int>& upSearchExtreme_vec, vector<int>& downSearchExtreme_vec,
		 const int windowNumber_perGroup,const int itemNum_perGroup,const int SC_Band){

	this->depressed_groupQuery_upSearchExtreme.resize(upSearchExtreme_vec.size());
	std::copy(upSearchExtreme_vec.begin(),upSearchExtreme_vec.end(),depressed_groupQuery_upSearchExtreme.begin());
	this->depressed_groupQuery_downSearchExtreme.resize(downSearchExtreme_vec.size());
	std::copy(downSearchExtreme_vec.begin(),downSearchExtreme_vec.end(),depressed_groupQuery_downSearchExtreme.begin());


	depressed_gpuManager.init_dimensionNum_perQuery(winDim);//specially note: to correct the dimensions of every query item


	this->sc_band = SC_Band;
	this->windowNum_perGroup = windowNumber_perGroup;
	this->groupQuery_item_num = itemNum_perGroup;

	init_dev_groupQuery_constantMem(windowNumber_perGroup,itemNum_perGroup);

	this->h_groupQuery_info_set.resize(groupQuery_info_set.size());
	thrust::copy(groupQuery_info_set.begin(),groupQuery_info_set.end(),this->h_groupQuery_info_set.begin());

	copy_GroupQueryInfo_vec_FromHostToDevice(h_groupQuery_info_set, this->d_groupQuery_info_set);

	this->groupQuery_maxdimension = this->getMaxDimensionsOfGroupQueries(groupQuery_info_set);
	this->depressed_d_groupQuery_unfinished.resize(groupQuery_info_set.size(),true);
	this->d_groupQuerySet_items_lowerBoundThreshold.resize(this->getTotalQueryItemNum(),0);

	this->d_groupQuerySet_lowerBound.resize(this->getTotalQueryItemNum()*this->depressed_getMaxWindowFeatureNumber()*winDim,INITIAL_LABEL_VALUE);//different equal classes (ECs) corresponding to different time series features



	//note: must configure h_groupQuery_info_set and groupQuery_maxdimension firstly
	depressed_conf_windowQuery_onGpuManager(sc_band);
	//vector<GpuQuery> windowQuerySet;
	//set_windowQueryInfo_set(sc_band, windowQuerySet);
	//gpuManager.init_GPU_query(windowQuerySet);
	//this->d_winQuery_lowerBound.resize(depressed_gpuManager.getTotalnumOfQuery()*depressed_gpuManager.getMaxFeatureID());

}


/**
 * TODO:
 * initialize and set query
 *
 *
 */
void TSGPUManager::conf_dev_query(vector<GroupQuery_info*>& groupQuery_info_set,
		 const int windowNumber_perGroup,const int itemNum_perGroup, const int winDim, const int sc_band){

	this->sc_band = sc_band;
	this->winDim = winDim;
	this->windowNum_perGroup = windowNumber_perGroup;
	this->groupQuery_item_num = itemNum_perGroup;
	this->maxWindowFeatureNumber = this->comp_maxWindowFeatureNumber();
	init_dev_groupQuery_constantMem(windowNumber_perGroup,itemNum_perGroup);
	this->h_groupQuery_info_set.resize(groupQuery_info_set.size());
	thrust::copy(groupQuery_info_set.begin(),groupQuery_info_set.end(),this->h_groupQuery_info_set.begin());

	copy_GroupQueryInfo_vec_FromHostToDevice(h_groupQuery_info_set, this->d_groupQuery_info_set);
	this->groupQuery_maxdimension = this->getMaxDimensionsOfGroupQueries(groupQuery_info_set);
	//this->depressed_d_groupQuery_unfinished.resize(groupQuery_info_set.size(),true);
	this->d_groupQuerySet_items_lowerBoundThreshold.resize(this->getTotalQueryItemNum(),0);

	this->d_groupQuerySet_lowerBound.resize(this->getTotalQueryItemNum()*this->getMaxFeatureNumber(),INITIAL_LABEL_VALUE);//different equal classes (ECs) corresponding to different time series features

	//note: must configure h_groupQuery_info_set  firstly

	conf_dev_windowQuery();
	if(enhancedLowerBound){
		dev_compute_TSData_upperLowerBound();//configure the upper and lower bound for blade data with sc_band, preparition for compute lower bound of windowQuery
	}

}


void TSGPUManager::depressed_conf_windowQuery_onGpuManager(int sc_band){
	vector<GpuWindowQuery> windowQuerySet;
	depressed_set_windowQueryInfo_set(sc_band, windowQuerySet);
	depressed_gpuManager.init_GPU_query(windowQuerySet);
	this->d_windowQuery_lowerBound.resize(depressed_gpuManager.getTotalnumOfQuery()*depressed_gpuManager.getMaxFeatureNumber());
}

/**
 * TODO:
 * configure the sliding window query information
 *
 * note: must configure h_groupQuery_info_set  firstly
 */
void TSGPUManager::depressed_conf_dev_windowQuery(){


	vector<GpuWindowQuery> windowQuerySet;
	set_windowQueryInfo_set(sc_band, windowQuerySet);

	host_vector<int> h_windowQuery_lowerBound_len(windowQuerySet.size(),0);
	h_windowQuery_info_set.reserve( windowQuerySet.size() );
	for(int i = 0; i < windowQuerySet.size(); i++)
	{
		WindowQueryInfo *queryInfo = new WindowQueryInfo(windowQuerySet[i]);
		h_windowQuery_info_set.push_back(queryInfo);
		h_windowQuery_lowerBound_len[i] = h_ts_blade_len[windowQuerySet[i].bladeId]/winDim;
	}

	d_windowQuery_info_set.reserve( h_windowQuery_info_set.size() );
	// copy queryInfo to gpu
	depressed_copy_windowQueryInfo_vec_fromHostToDevice( h_windowQuery_info_set, d_windowQuery_info_set );


	this->d_windowQuery_lowerBound_endIdx=h_windowQuery_lowerBound_len;
	thrust::inclusive_scan(d_windowQuery_lowerBound_endIdx.begin(),d_windowQuery_lowerBound_endIdx.end(),d_windowQuery_lowerBound_endIdx.begin());
	d_windowQuery_lowerBound.resize(d_windowQuery_lowerBound_endIdx.back(),0);


	init_dev_windowQuery_constantMem(winDim, this->maxWindowFeatureNumber);


}


/**
 * TODO:
 * configure the sliding window query information
 *
 * note: must configure h_groupQuery_info_set  firstly
 */
void TSGPUManager::conf_dev_windowQuery(){

	vector<GpuWindowQuery> windowQuerySet;
	set_windowQueryInfo_set(sc_band, windowQuerySet);

	host_vector<int> h_windowQuery_lowerBound_len(windowQuerySet.size(),0);

	h_windowQuery_info_set.reserve( windowQuerySet.size() );
	for(int i = 0; i < windowQuerySet.size(); i++)
	{
		WindowQueryInfo *queryInfo = new WindowQueryInfo(windowQuerySet[i]);
		h_windowQuery_info_set.push_back(queryInfo);
		h_windowQuery_lowerBound_len[i] = h_ts_blade_len[windowQuerySet[i].bladeId]/winDim;
	}


	d_windowQuery_info_set.reserve(h_windowQuery_info_set.size());
	// copy queryInfo to gpu
	copy_windowQueryInfo_vec_fromHostToDevice(h_windowQuery_info_set, d_windowQuery_info_set);
	//denote all windowQuery to be verified for lowerbound
	this->d_windowQuery_LBKeogh_updatedLabel = h_windowQuery_LBKeogh_updatedLabel;




	this->d_windowQuery_lowerBound_endIdx=h_windowQuery_lowerBound_len;
	thrust::inclusive_scan(d_windowQuery_lowerBound_endIdx.begin(),d_windowQuery_lowerBound_endIdx.end(),d_windowQuery_lowerBound_endIdx.begin());
	d_windowQuery_lowerBound.resize(d_windowQuery_lowerBound_endIdx.back(),0);
	if(enhancedLowerBound){
		d_windowQuery_lowerBound_q2d.resize(d_windowQuery_lowerBound_endIdx.back(),0);
	}



	init_dev_windowQuery_constantMem(winDim, this->maxWindowFeatureNumber);


}

/**
 * when the inverted index is built by bucket unit, use this struct
 */
void TSGPUManager::depressed_dev_BidirectionExpansion_bucketUnit(){
	depressed_gpuManager.dev_BidirectionExpansion(
			DataToIndex_keywordMap_bucketUnit(),
			IndexToData_lastPosMap_bucketUnit(),
			Lp_distance());
}

/**
 * when the inverted index is built by bucket with width, use this struct
 */
void TSGPUManager::depressed_dev_BidirectionExpansion_bucket_exclusive(){
	depressed_gpuManager.dev_BidirectionExpansion(
			DataToIndex_keywordMap_bucket(),
			IndexToData_lastPosMap_bucket_exclusive(),
			Lp_distance());
}

/**
 * 1.after bi-expansion, compute the lower bound for each window queries.
 * 2. should be followed by the function dev_BidirectionExpansion_bucketUnit();
 * 3.  when the inverted index is built by bucket unit, use this struct
 */
void TSGPUManager::depressed_dev_compute_windowQuery_LowerBound_bucketUnit(){

	//with debug purpose
	//print_windowQuery_LowerBound();

	depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl&&(!(this->depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled));

	depressed_compute_windowQuery_LowerBound_template<<< depressed_gpuManager.getTotalnumOfQuery(),THREAD_PER_BLK, (2*winDim +sc_band)*sizeof(float)>>>(
			raw_pointer_cast(depressed_gpuManager.get_d_query_feature_reference().data()),
			raw_pointer_cast(depressed_gpuManager.get_d_query_info_reference().data()), // remember the expansion position for each window query
			raw_pointer_cast(depressed_gpuManager.get_d_indexDimensionEntry_vec_reference().data()),
			raw_pointer_cast(this->d_windowQuery_lowerBound.data()),//record the lower bound of each window query
			this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl,
			this->sc_band,
			raw_pointer_cast(this->depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec.data()),
			raw_pointer_cast(this->depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec.data()),
			DataToIndex_keywordMap_bucketUnit(),
			IndexToData_lastPosMap_bucketUnit(),
			Lp_distance()
			);

	this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = false;//the update of ACD only need to do once for all topk query of this step
}


/**
 * 1.after bi-expansion, compute the lower bound for each window queries.
 * 2. should be followed by the function dev_BidirectionExpansion_bucket_exclusive();
 * 3.  when the inverted index is built by bucket with width, use this struct
 */
void TSGPUManager::depressed_dev_compute_windowQuery_LowerBound_bucket_inclusive(){

	//with debug purpose
	//print_windowQuery_LowerBound();

	depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl&&(!(this->depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled));

	depressed_compute_windowQuery_LowerBound_template<<< depressed_gpuManager.getTotalnumOfQuery(),THREAD_PER_BLK, (2*winDim +sc_band)*sizeof(float)>>>(
			raw_pointer_cast(depressed_gpuManager.get_d_query_feature_reference().data()),
			raw_pointer_cast(depressed_gpuManager.get_d_query_info_reference().data()), // remember the expansion position for each window query
			raw_pointer_cast(depressed_gpuManager.get_d_indexDimensionEntry_vec_reference().data()),
			raw_pointer_cast(this->d_windowQuery_lowerBound.data()),//record the lower bound of each window query
			this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl,
			this->sc_band,
			raw_pointer_cast(this->depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec.data()),
			raw_pointer_cast(this->depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec.data()),
			DataToIndex_keywordMap_bucket(),
			IndexToData_lastPosMap_bucket_inclusive(),
			Lp_distance()
			);

	this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = false;//the update of ACD only need to do once for all topk query of this step
}

/**
 * TODO:
 *    compute the lower bound for each window query
 */
void TSGPUManager::dev_compute_windowQuery_lowerBound(){

	int windowQueryNum = d_windowQuery_info_set.size();
	compute_windowQuery_lowerBound<<< windowQueryNum*this->winDim, THREAD_PER_BLK >>>(
		raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
		raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //the end idx for each blades (i.e. the boundary of different blades)
		raw_pointer_cast(this->d_windowQuery_info_set.data()), // input: records each window query
		raw_pointer_cast(this->d_windowQuery_lowerBound.data()),//output: record the lower bound and upper bound of each window query
		raw_pointer_cast(this->d_windowQuery_lowerBound_endIdx.data()), //input: record the end idx for each windowQuery
		raw_pointer_cast(this->d_windowQuery_LBKeogh_updatedLabel.data()),
		this->winDim, this->sc_band,
		LBKeogh_L2_distance());

	//with debug purpose
	//cout<<"with debug purpose dev_compute_windowQuery_lowerBound()"<<endl;
	//print_windowQuery_lowerBound();

}

/**
 * TODO:
 *    compute the lower bound for each window query
 */
void TSGPUManager::dev_compute_windowQuery_enhancedLowerBound(){

		int windowQueryNum = d_windowQuery_info_set.size();
		compute_windowQuery_enhancedLowerBound<<< windowQueryNum*this->winDim, THREAD_PER_BLK >>>(
			raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
			raw_pointer_cast(this->d_ts_data_u.data()),
			raw_pointer_cast(this->d_ts_data_l.data()),
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //the end idx for each blades (i.e. the boundary of different blades)
			raw_pointer_cast(this->d_windowQuery_info_set.data()), // input: records each window query
			raw_pointer_cast(this->d_windowQuery_lowerBound.data()),//output: record the lower bound and upper bound of each window query (d2q)
			raw_pointer_cast(this->d_windowQuery_lowerBound_q2d.data()),//output: record the lower bound and upper bound of each window query (d2q)
			raw_pointer_cast(this->d_windowQuery_lowerBound_endIdx.data()), //input: record the end idx for each windowQuery
			raw_pointer_cast(this->d_windowQuery_LBKeogh_updatedLabel.data()),
			this->winDim, this->sc_band, this->enhancedLowerBound,
			LBKeogh_L2_distance());



		//with debug purpose
		//cout<<"with debug purpose dev_compute_windowQuery_lowerBound()"<<endl;
		//print_windowQuery_lowerBound();
}


void TSGPUManager::dev_compute_TSData_upperLowerBound(){

	compute_tsData_UpperLowerBound<<<this->getBladeNum(),THREAD_PER_BLK>>>(
		raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
		raw_pointer_cast(this->d_ts_data_u.data()), //output: record the lower and upper bound for ts_data
		raw_pointer_cast(this->d_ts_data_l.data()),
		raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //the end idx for each blades (i.e. the boundary of different blades)
		this->sc_band
		);
}


/**
 * 1. compute the lower bound for each group query based on the lower bound of window queries
 * 2. should be followed before dev_compute_windowQuery_LowerBound();
 */
void TSGPUManager::depressed_dev_compute_groupQuery_LowerBound(){



		int total_ec_num = this->getGroupQueryNum()*winDim;// each item query has winDim number of equal class

		depressed_compute_groupQuery_LowerBound<<<total_ec_num, THREAD_PER_BLK,this->getQueryItemsNumberPerGroup()*sizeof(uint)*2 >>>(
				raw_pointer_cast(this->d_windowQuery_lowerBound.data()), //input: record the lower bound and upper bound of each window query
				raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:
				raw_pointer_cast(this->depressed_d_groupQuery_unfinished.data()), //input: record the status of this group query
				raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()) //output: record the lower bound of each group query
		);

		//with debug purpose
		//print_groupQuery_LowerBound();
}


void TSGPUManager::dev_compute_groupQuery_lowerBound(){

		int total_ec_num = this->getGroupQueryNum()*winDim;// each item query has winDim number of equal class

		compute_groupQuery_lowerBound<<<total_ec_num, THREAD_PER_BLK,this->getQueryItemsNumberPerGroup()*sizeof(uint) >>>(
				raw_pointer_cast(this->d_windowQuery_lowerBound.data()), //input: record the lower bound and upper bound of each window query
				raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:
				raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()) //output: record the lower bound of each group query
				);

		//with debug purpose
		//cout<<"with debug purpose dev_compute_groupQuery_lowerBound() "<<endl;
		//print_groupQuery_LowerBound();
}


void TSGPUManager::dev_compute_groupQuery_enhancedLowerBound(){
	int total_ec_num = this->getGroupQueryNum()*winDim;// each item query has winDim number of equal class

		compute_groupQuery_enhancedLowerBound<<<total_ec_num, THREAD_PER_BLK,this->getQueryItemsNumberPerGroup()*sizeof(uint) >>>(
				raw_pointer_cast(this->d_windowQuery_lowerBound.data()), //input: record the lower bound and upper bound of each window query
				raw_pointer_cast(this->d_windowQuery_lowerBound_q2d.data()),
				raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:
				raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //output: record the lower bound of each group query
				this->enhancedLowerBound_sel
				);


		//cout<<"with debug purpose: print dev_compute_groupQuery_enhancedLowerBound():print_groupQuery_LowerBound()"<<endl;
		//with debug purpose
		//this->print_groupQuery_LowerBound();

}
/**
 * TODO:
 * scan the this->d_groupQuerySet_lowerBound table, select the candidates whose lower bound is larger than the lower bound threshold
 * If there is no candidate for this query, also update the query status as finished
 *
 * output:
 * _d_groupQuerySet_candidates: the candidates whose lower bound is larger than the threshold
 */
void TSGPUManager::depressed_dev_output_groupQuery_candidates(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//output
		device_vector<int>& d_groupQuerySet_candidates_size//output
	){//output


	d_groupQuerySet_candidates.clear();
	//d_groupQuerySet_candidates.shrink_to_fit();
	d_groupQuerySet_candidates_size.clear();
	//d_groupQuerySet_candidates_endIdx.shrink_to_fit();


	int totalQueryItemNum = this->getTotalQueryItemNum();

	device_vector<int> d_thread_threshold_endIdx(totalQueryItemNum*THREAD_PER_BLK, 0);// prefix count for each thread
	d_groupQuerySet_candidates_size.resize(totalQueryItemNum,0);// end idx for each group query in d_groupQuerySet_candidates


	 depressed_prefixCount_groupQueryCandidates_Threshold<<<totalQueryItemNum,THREAD_PER_BLK, sizeof(int)*THREAD_PER_BLK>>>(
			raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //input: record the lower bound of each group query
			raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data()), //input: threshold for each group query
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
			raw_pointer_cast(d_thread_threshold_endIdx.data()), //output: compute candidates per thread
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()), //output: compute candidates per block, later use inclusive sum to get address
			raw_pointer_cast(this->d_groupQuery_info_set.data())//output:check the query status. if there is no candidates for this query item, label this query item as finished
			);


	 thrust::inclusive_scan(d_thread_threshold_endIdx.begin(), d_thread_threshold_endIdx.end(),
			 d_thread_threshold_endIdx.begin()); // per thread inclusive scan, to get end position to store candidates of each thread

	 //thrust::inclusive_scan(d_groupQuerySet_candidates_endIdx.begin(), d_groupQuerySet_candidates_endIdx.end(),
		//	 d_groupQuerySet_candidates_endIdx.begin()); // per block inclusive scan, to get end position to store candidates of each block

	 d_groupQuerySet_candidates.resize(d_thread_threshold_endIdx[d_thread_threshold_endIdx.size()-1]);

	 depressed_output_groupQueryCandidates_Threshold<<<totalQueryItemNum,THREAD_PER_BLK>>>(
			 raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //input: record the lower bound and upper bound of each group query
			 raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data()),//input: threshold for each group query
			 raw_pointer_cast(this->d_groupQuery_info_set.data()),//input:check wheter this group query (item) is finished
			 raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
			 raw_pointer_cast(d_thread_threshold_endIdx.data()),  //input: end idx for each threadIdx and each block
			 raw_pointer_cast(d_groupQuerySet_candidates.data())//output: store the candidates for all group queries
	 		);

}



/**
 * TODO:
 * scan the this->d_groupQuerySet_lowerBound table, select the candidates whose lower bound is larger than the lower bound threshold *
 * output:
 * _d_groupQuerySet_candidates: the candidates whose lower bound is larger than the threshold
 */
void TSGPUManager::dev_output_groupQuery_candidates(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//output
		device_vector<int>& d_groupQuerySet_candidates_size//output
	){//output


	d_groupQuerySet_candidates.clear();
	//d_groupQuerySet_candidates.shrink_to_fit();
	d_groupQuerySet_candidates_size.clear();
	//d_groupQuerySet_candidates_endIdx.shrink_to_fit();


	int totalQueryItemNum = this->getTotalQueryItemNum();

	device_vector<int> d_thread_threshold_endIdx(totalQueryItemNum*THREAD_PER_BLK, 0);// prefix count for each thread
	d_groupQuerySet_candidates_size.resize(totalQueryItemNum,0);// end idx for each group query in d_groupQuerySet_candidates


	 prefixCount_groupQueryCandidates_Threshold<<<totalQueryItemNum,THREAD_PER_BLK, sizeof(int)*THREAD_PER_BLK>>>(
			raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //input: record the lower bound of each group query
			raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data()), //input: threshold for each group query
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
			raw_pointer_cast(d_thread_threshold_endIdx.data()), //output: compute candidates per thread
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()), //output: compute candidates per block, later use inclusive sum to get address
			raw_pointer_cast(this->d_groupQuery_info_set.data())//output:check the query status. if there is no candidates for this query item, label this query item as finished
			);


	 thrust::inclusive_scan(d_thread_threshold_endIdx.begin(), d_thread_threshold_endIdx.end(),
			 d_thread_threshold_endIdx.begin()); // per thread inclusive scan, to get end position to store candidates of each thread



	 //thrust::inclusive_scan(d_groupQuerySet_candidates_endIdx.begin(), d_groupQuerySet_candidates_endIdx.end(),
		//	 d_groupQuerySet_candidates_endIdx.begin()); // per block inclusive scan, to get end position to store candidates of each block

	 d_groupQuerySet_candidates.resize(d_thread_threshold_endIdx.back());

	 output_groupQueryCandidates_Threshold<<<totalQueryItemNum,THREAD_PER_BLK>>>(
			 raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //input: record the lower bound and upper bound of each group query
			 raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data()),//input: threshold for each group query
			 raw_pointer_cast(this->d_groupQuery_info_set.data()),//input:check wheter this group query (item) is finished
			 raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
			 raw_pointer_cast(d_thread_threshold_endIdx.data()),  //input: end idx for each threadIdx and each block
			 raw_pointer_cast(d_groupQuerySet_candidates.data())//output: store the candidates for all group queries
	 		);


}




/**
 * TODO:
 * scan the this->d_groupQuerySet_lowerBound table, select the candidates whose lower bound is larger than the lower bound threshold
 *  *
 * output:
 * _d_groupQuerySet_candidates: the candidates whose lower bound is larger than the threshold
 */
void TSGPUManager::dev_output_groupQuery_candidates(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//output
		device_vector<int>& d_groupQuerySet_candidates_size,//output
		device_vector<int>& d_thread_threshold_endIdx
	){//output,


	d_groupQuerySet_candidates.clear();
	//d_groupQuerySet_candidates.shrink_to_fit();
	d_groupQuerySet_candidates_size.clear();
	//d_groupQuerySet_candidates_endIdx.shrink_to_fit();


	int totalQueryItemNum = this->getTotalQueryItemNum();

	d_thread_threshold_endIdx.resize(totalQueryItemNum*THREAD_PER_BLK, 0);// prefix count for each thread
	d_groupQuerySet_candidates_size.resize(totalQueryItemNum,0);// end idx for each group query in d_groupQuerySet_candidates


	 prefixCount_groupQueryCandidates_Threshold<<<totalQueryItemNum,THREAD_PER_BLK, sizeof(int)*THREAD_PER_BLK>>>(
			raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //input: record the lower bound of each group query
			raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data()), //input: threshold for each group query
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
			raw_pointer_cast(d_thread_threshold_endIdx.data()), //output: compute candidates per thread
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()), //output: compute candidates per block, later use inclusive sum to get address
			raw_pointer_cast(this->d_groupQuery_info_set.data())//output:check the query status. if there is no candidates for this query item, label this query item as finished
			);

	 //for exp
	// cout<<"for exp: print mean and std_var of un-filtered id per thread"<<endl;
	// print_d_groupQuerySet_candidates_scan_size( d_thread_threshold_endIdx);
	//end for exp

	 thrust::inclusive_scan(d_thread_threshold_endIdx.begin(), d_thread_threshold_endIdx.end(),
			 d_thread_threshold_endIdx.begin()); // per thread inclusive scan, to get end position to store candidates of each thread



	 //thrust::inclusive_scan(d_groupQuerySet_candidates_endIdx.begin(), d_groupQuerySet_candidates_endIdx.end(),
		//	 d_groupQuerySet_candidates_endIdx.begin()); // per block inclusive scan, to get end position to store candidates of each block

	 d_groupQuerySet_candidates.resize(d_thread_threshold_endIdx.back());

	 output_groupQueryCandidates_Threshold<<<totalQueryItemNum,THREAD_PER_BLK>>>(
			 raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //input: record the lower bound and upper bound of each group query
			 raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data()),//input: threshold for each group query
			 raw_pointer_cast(this->d_groupQuery_info_set.data()),//input:check wheter this group query (item) is finished
			 raw_pointer_cast(this->d_ts_data_blade_endIdx.data()), //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
			 raw_pointer_cast(d_thread_threshold_endIdx.data()),  //input: end idx for each threadIdx and each block
			 raw_pointer_cast(d_groupQuerySet_candidates.data())//output: store the candidates for all group queries
	 		);


}

/**
 * TODO:
 * 		verify the candidates (i.e. caculate the true distance of entry in d_groupQuerySet_candidates) and the same time label the entry as verified in d_groupQuerySet_lowerBound
 */
void TSGPUManager::depressed_dev_scan_verifyCandidates_fixedNum(
		int verified_candidates_num,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
		device_vector<int>& d_groupQuerySet_candidates_size){

	//int totalQueries = (this->getGroupQueryNum()) * (this->getGroupQueryItemsNumber());

	depressed_scan_verifyCandidates<<<this->getTotalQueryItemNum(),THREAD_PER_BLK,(this->getGroupQueryMaxDimension()+1)*sizeof(float)>>>(
			this->sc_band,//input: Sakoe-Chiba Band
			verified_candidates_num,
			raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:group query
			raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()),//the endidx for each blades (i.e. the boundary of different blades)
			raw_pointer_cast(d_groupQuerySet_candidates.data()),//input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates_dist
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()),//record the idx for each group query in d_groupQuerySet_Candidates
			raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()) //output: update the boundEntry after verification and set the verified (assign -1 to entry)
		);

}

void TSGPUManager::depressed_dev_scan_verifyCandidate(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
		device_vector<int>& d_groupQuerySet_candidates_endIdx){

		depressed_scan_verifyCandidates<<<this->getTotalQueryItemNum(),THREAD_PER_BLK,(this->getGroupQueryMaxDimension()+1)*sizeof(float)>>>(
				this->sc_band,//input: Sakoe-Chiba Band
				0,
				raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:group query
				raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
				raw_pointer_cast(this->d_ts_data_blade_endIdx.data()),//the endidx for each blades (i.e. the boundary of different blades)
				raw_pointer_cast(d_groupQuerySet_candidates.data()),//input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates_dist
				raw_pointer_cast(d_groupQuerySet_candidates_endIdx.data()),//record the idx for each group query in d_groupQuerySet_Candidates
				raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()) //output: update the boundEntry after verification and set the verified (assign -1 to entry)
			);

}



/**
 * TODO:
 * 		verify the candidates (i.e. caculate the true distance of entry in d_groupQuerySet_candidates) and the same time label the entry as verified in d_groupQuerySet_lowerBound
 */
void TSGPUManager::dev_scan_verifyCandidates_fixedNum(
		int verified_candidates_num,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
		device_vector<int>& d_groupQuerySet_candidates_size){

	//int totalQueries = (this->getGroupQueryNum()) * (this->getGroupQueryItemsNumber());

	scan_verifyCandidates<<<this->getTotalQueryItemNum(),THREAD_PER_BLK,(this->getGroupQueryMaxDimension()+1)*sizeof(float)>>>(
			this->sc_band,//input: Sakoe-Chiba Band
			verified_candidates_num,
			raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:group query
			raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()),//the endidx for each blades (i.e. the boundary of different blades)
			raw_pointer_cast(d_groupQuerySet_candidates.data()),//input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates_dist
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()),//record the idx for each group query in d_groupQuerySet_Candidates
			//raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //output: update the boundEntry after verification and set the verified (assign -1 to entry)
			Dtw_SCBand_Func_modulus_flt(this->sc_band)
	);

}

void TSGPUManager::dev_scan_verifyCandidate(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
		device_vector<int>& d_groupQuerySet_candidates_endIdx){

		scan_verifyCandidates<<<this->getTotalQueryItemNum(),THREAD_PER_BLK,(this->getGroupQueryMaxDimension()+1)*sizeof(float)>>>(
				this->sc_band,//input: Sakoe-Chiba Band
				0,
				raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:group query
				raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
				raw_pointer_cast(this->d_ts_data_blade_endIdx.data()),//the endidx for each blades (i.e. the boundary of different blades)
				raw_pointer_cast(d_groupQuerySet_candidates.data()),//input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates_dist
				raw_pointer_cast(d_groupQuerySet_candidates_endIdx.data()),//record the idx for each group query in d_groupQuerySet_Candidates
				//raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()), //output: update the boundEntry after verification and set the verified (assign -1 to entry)
				Dtw_SCBand_Func_modulus_flt(this->sc_band)
		);
}

/**
 * TODO:
 * scan the candidate to verify. Use one kernel to deal with the candidate scanned by each thread
 */
void TSGPUManager::dev_scan_verifyCandidate_perThreadGenerated(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
		device_vector<int>& d_groupQuerySet_candidate_threadThreshold_endIdx
		){

	int mergeThreadInterval = THREAD_PER_BLK/2;//

	scan_verifyCandidates_perThreadGenerated<<<this->getTotalQueryItemNum()*(THREAD_PER_BLK/mergeThreadInterval),THREAD_PER_BLK,(this->getGroupQueryMaxDimension()+1)*sizeof(float)>>>(
			this->sc_band, //input: Sakoe-Chiba Band
			raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:group query
			raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()),//the endidx for each blades (i.e. the boundary of different blades)
			raw_pointer_cast(d_groupQuerySet_candidates.data()),//input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates_dist
			raw_pointer_cast(d_groupQuerySet_candidate_threadThreshold_endIdx.data()),//record the idx for each group query in d_groupQuerySet_Candidates
			THREAD_PER_BLK,mergeThreadInterval,
			Dtw_SCBand_Func_modulus_flt(this->sc_band));

}

//==================start the code with random selection



/**
 * TODO:
 * do topk query under DTW, for each step of verification, we randomly select "verify_candidates_num" to get a bound for topk query
 */
void TSGPUManager::depressed_exact_TopK_query_DTW_randomSelect(int topk, bool isBucketUnit){

	cudaProfilerStart();
	cudaEvent_t start, stop;

	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEvent_t seg_start, seg_stop;
	cudaEventCreate(&seg_start);
	cudaEventCreate(&seg_stop);

	cudaEventRecord(start, 0);


	bool finished = false;
	int verify_candidates_num = THREAD_PER_BLK;
	this->d_groupQuerySet_topkResults.clear();
	this->d_groupQuerySet_topkResults.resize(this->getTotalQueryItemNum() * topk);//every query item with top-k results

	// data flow:
    //d_groupQuerySet_candidates_scan => d_groupQuerySet_candidates_selected => d_groupQuerySet_candidates_topk => this->d_groupQuerySet_topkResults
	//d_groupQuerySet_candidates_scan is flexible size for each query item, *_selected and *_topk is fixed size for each query item
	device_vector<CandidateEntry> d_groupQuerySet_candidates_scan;
	device_vector<int> d_groupQuerySet_candidates_scan_size;

	device_vector<CandidateEntry> d_groupQuerySet_candidates_verified;
	device_vector<int> d_groupQuerySet_candidates_verified_size;

	//random select topk elements to get a initial threshold for topk query
	//improve here !!! this method can be replaced by fast k-selection of the first k candidates with smallest lower bound
	cudaEventRecord(seg_start,0);
	depressed_init_topkThreshold_randomSelect(topk);
	cudaEventRecord(seg_stop,0);
	cudaEventSynchronize(seg_stop);
	cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);



	float exec_time_seg[6]={0};

	int iteration = 0;
	while(!finished){
		cout<<"topk-query iteration:"<<iteration<<endl;
		iteration++;

		//step 1
		cudaEventRecord(seg_start,0);
		if(isBucketUnit){
			depressed_dev_BidirectionExpansion_bucketUnit();
			depressed_dev_compute_windowQuery_LowerBound_bucketUnit();
		}else{
			depressed_dev_BidirectionExpansion_bucket_exclusive();
			depressed_dev_compute_windowQuery_LowerBound_bucket_inclusive();
		}
		cudaEventRecord(seg_stop,0);
		cudaEventSynchronize(seg_stop);
		cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);
		exec_time_seg[0]+=elapsedTime;

		cudaEventRecord(seg_start,0);
		depressed_dev_compute_groupQuery_LowerBound();
		cudaEventRecord(seg_stop,0);
		cudaEventSynchronize(seg_stop);
		cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);
		exec_time_seg[1]+=elapsedTime;

		cudaEventRecord(seg_start,0);
		depressed_dev_output_groupQuery_candidates(d_groupQuerySet_candidates_scan,d_groupQuerySet_candidates_scan_size);
		cudaEventRecord(seg_stop,0);
		cudaEventSynchronize(seg_stop);
		cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);
		exec_time_seg[2]+=elapsedTime;

		cudaEventRecord(seg_start,0);
		finished = dev_check_groupQuery_finished();
		cudaEventRecord(seg_stop,0);
		cudaEventSynchronize(seg_stop);
		cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);
		exec_time_seg[3]+=elapsedTime;


		cout<<" finished ="<<finished<<" with iteration:"<<iteration<<endl;
		if(finished) break;

		device_vector<int> d_groupQuerySet_candidates_scan_endIdx(d_groupQuerySet_candidates_scan_size.size());
		thrust::inclusive_scan(d_groupQuerySet_candidates_scan_size.begin(),d_groupQuerySet_candidates_scan_size.end(),d_groupQuerySet_candidates_scan_endIdx.begin());
		cudaEventRecord(seg_start,0);
		depressed_dev_randomSelect_and_Verify_topkCandidates(
				topk,
				verify_candidates_num,
				d_groupQuerySet_candidates_scan,
				d_groupQuerySet_candidates_scan_endIdx,
				d_groupQuerySet_candidates_verified,
				d_groupQuerySet_candidates_verified_size
				);
		cudaEventRecord(seg_stop,0);
		cudaEventSynchronize(seg_stop);
		cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);
		exec_time_seg[4]+=elapsedTime;
		//with debug purpose
		//print_d_groupQuerySet_candidates_scan(d_groupQuerySet_candidates_scan,
		//d_groupQuerySet_candidates_scan_endIdx);


		cudaEventRecord(seg_start,0);
		dev_maintain_topkCandidates_mergeSort(
			topk,
			d_groupQuerySet_candidates_verified,
			d_groupQuerySet_candidates_verified_size);
		cudaEventRecord(seg_stop,0);
		cudaEventSynchronize(seg_stop);
		cudaEventElapsedTime(&elapsedTime, seg_start, seg_stop);
		exec_time_seg[5]+=elapsedTime;
		//with debug purpose
		//print_d_groupQuerySet_topkResults();

	}


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);




}




/**
 * TODO:
 * in the first step (initialized step), we randomly select k elements for the candidates to get a bound for the top-k query
 *
 * int candidates_num: assumption: it is better to equal to the thread number ?  improve here !!!
 */
void TSGPUManager::depressed_dev_randomSelect_verify_candidates(
		int topk,
		int candidates_num,
		device_vector<CandidateEntry>& groupQuerySet_candidates_selected
		){

	int totalQueyItemNum = this->getTotalQueryItemNum();//(this->getGroupQueryNum()) * (this->getGroupQueryItemsNumber());

	//random select candidates
	host_vector<CandidateEntry> h_groupQuerySet_candidates(totalQueyItemNum*candidates_num);
	host_vector<int> h_groupQuerySet_candidates_size(totalQueyItemNum,candidates_num);//here!

	for(uint i=0;i<totalQueyItemNum;i++){

		int gid = i/groupQuery_item_num;
		int gq_item_id = i%groupQuery_item_num;
		int blade_id = h_groupQuery_info_set[gid]->blade_id;
		int len = h_ts_blade_len[blade_id]- h_groupQuery_info_set[gid]->getItemDimension(gq_item_id)+1;

		int interval = ((len)/candidates_num!= 0) ? (len/candidates_num):1;
		int actual_candidates_num = 0;
		for(uint j=0;j<candidates_num;j++){
			int fid = interval*j;//+random()%interval;
			if(fid<len){
			h_groupQuerySet_candidates[i*candidates_num+j].feature_id = fid;
			actual_candidates_num++;
			}
		}
		h_groupQuerySet_candidates_size[i] = actual_candidates_num;
	}
	device_vector<CandidateEntry> d_groupQuerySet_candidates = h_groupQuerySet_candidates;
	device_vector<int> d_groupQuerySet_candidates_size(h_groupQuerySet_candidates_size);

	//thrust::inclusive_scan(d_groupQuerySet_candidates_endIdx.begin(), d_groupQuerySet_candidates_endIdx.end(),
	//			 d_groupQuerySet_candidates_endIdx.begin()); // per block inclusive scan


	depressed_scan_verifyCandidates<<<totalQueyItemNum,THREAD_PER_BLK,(this->getGroupQueryMaxDimension()+1)*sizeof(float)>>>(
			this->sc_band,//input: Sakoe-Chiba Band
			candidates_num,//if candidats_number == 0, the candidates number is flexible, else all of query items are within the same or smaller number of candidates
			raw_pointer_cast(this->d_groupQuery_info_set.data()), //input:group query
			raw_pointer_cast(this->d_ts_data.data()), //the time series blade, note: there may be multiple blades
			raw_pointer_cast(this->d_ts_data_blade_endIdx.data()),//the endidx for each blades (i.e. the boundary of different blades)
			raw_pointer_cast(d_groupQuerySet_candidates.data()),//input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()),//input: record the actual size for each group query in d_groupQuerySet_Candidates
			raw_pointer_cast(this->d_groupQuerySet_lowerBound.data()) //output: update the boundEntry after verification and set the bound as -1 to indicate this entry has been verified
			);


	device_vector<int> groupQuerySet_candidates_selected_size;//improve here!!! temporary vector, not used

	this->dev_select_candidates_mergeSort_fixedNum(
			topk,
			candidates_num,
			d_groupQuerySet_candidates,
			d_groupQuerySet_candidates_size,
			groupQuerySet_candidates_selected,
			groupQuerySet_candidates_selected_size);
}

/**
 * TODO:
 * select selected_candidates_num candidates from the d_groupQuerySet_candidates by ascending order of distance
 * note: 1. input: int fixed_candidates_num,//number of items for selection, the actual number may be samller than fixed_candidates_num, which is recorded in "d_groupQuerySet_candidates_size"
 *       2. number of candidates should be able to cached in the shared memory
 */
void TSGPUManager::dev_select_candidates_mergeSort_fixedNum(
		int selected_candidates_num, //number of candidates want to select
		int fixed_candidates_num,//number of items for selection, the actual number may be samller than fixed_candidates_num, which is recorded in "d_groupQuerySet_candidates_size"
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input
		device_vector<int>& d_groupQuerySet_candidates_size,//input
		device_vector<CandidateEntry>& groupQuerySet_candidates_selected,//output
		device_vector<int>& groupQuerySet_candidates_selected_size//output:note this is size, not endIdx
		){

	int totalQueryItemNum = this->getTotalQueryItemNum();//(this->getGroupQueryNum()) * (this->getGroupQueryItemsNumber());

	groupQuerySet_candidates_selected.clear();
	groupQuerySet_candidates_selected.resize(totalQueryItemNum*selected_candidates_num);

	groupQuerySet_candidates_selected_size.clear();
	groupQuerySet_candidates_selected_size.resize(totalQueryItemNum,0);



	select_candidates_mergeSort<<<totalQueryItemNum,THREAD_PER_BLK, 2*fixed_candidates_num*sizeof(CandidateEntry)>>>(
			selected_candidates_num,
			fixed_candidates_num,
			fixed_candidates_num, //to allocate the size of shared memory
			raw_pointer_cast(d_groupQuerySet_candidates.data()),//input
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()),//record the idx for each group query in d_groupQuerySet_Candidates
			raw_pointer_cast(groupQuerySet_candidates_selected.data()),//output: result are stored here
			raw_pointer_cast(groupQuerySet_candidates_selected_size.data())//may be smaller than selected_candidates_num, which is recorded here
			);


}



/**
 * TODO:
 * select selected_candidates_num candidates from the d_groupQuerySet_candidates by ascending order of distance
 * note: 1. input: int fixed_candidates_num,//number of items for selection, the actual number may be samller than fixed_candidates_num, which is recorded in "d_groupQuerySet_candidates_size"
 *       2. number of candidates should be able to cached in the shared memory
 */
void TSGPUManager::dev_select_candidates_mergeSort_inPlace(
		int selected_candidates_num, //number of candidates want to select
		int fixed_candidates_num,//number of items for selection, the actual number may be samller than fixed_candidates_num, which is recorded in "d_groupQuerySet_candidates_size"
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input
		device_vector<int>& d_groupQuerySet_candidates_size,//input
		device_vector<CandidateEntry>& groupQuerySet_candidates_selected,//output
		device_vector<int>& groupQuerySet_candidates_selected_size//output:note this is size, not endIdx
		){

	int totalQueryItemNum = this->getTotalQueryItemNum();//(this->getGroupQueryNum()) * (this->getGroupQueryItemsNumber());

	select_candidates_mergeSort<<<totalQueryItemNum,THREAD_PER_BLK, 2*fixed_candidates_num*sizeof(CandidateEntry)>>>(
			selected_candidates_num,
			fixed_candidates_num,
			fixed_candidates_num, //to allocate the size of shared memory
			raw_pointer_cast(d_groupQuerySet_candidates.data()),//input
			raw_pointer_cast(d_groupQuerySet_candidates_size.data()),//record the idx for each group query in d_groupQuerySet_Candidates
			raw_pointer_cast(groupQuerySet_candidates_selected.data()),//output: result are stored here
			raw_pointer_cast(groupQuerySet_candidates_selected_size.data())//may be smaller than selected_candidates_num, which is recorded here
			);

}





/**
 * TODO:
 *     randomly select "selected_candidates_num" candidateEntry from d_groupQuerySet_candidates to groupQuerySet_candidates_kSelected
 */
void TSGPUManager::dev_select_candidates_random(
		int selectedCandidates_num,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,
		device_vector<int>& d_groupQuerySet_candidates_endIdx,
		device_vector<CandidateEntry>& groupQuerySet_candidates_selected,//output
		device_vector<int>& groupQuerySet_candidates_selected_size//output
		){


	int totalQueries = this->getTotalQueryItemNum();// (getGroupQueryNum()) * (getGroupQueryItemsNumber());

	//every query item has at most "selectedCandidates_num" items,
	//if candidates is less than that, the following part is padded with empty
	groupQuerySet_candidates_selected.clear();
	groupQuerySet_candidates_selected.resize(selectedCandidates_num*totalQueries);
	groupQuerySet_candidates_selected.shrink_to_fit();

	groupQuerySet_candidates_selected_size.clear();
	groupQuerySet_candidates_selected_size.resize(totalQueries,0);
	groupQuerySet_candidates_selected_size.shrink_to_fit();

	select_candidates_random<<<totalQueries,THREAD_PER_BLK>>>(
			selectedCandidates_num,
			raw_pointer_cast(d_groupQuerySet_candidates.data()),
			raw_pointer_cast(d_groupQuerySet_candidates_endIdx.data()),//record the idx for each group query in d_groupQuerySet_Candidates
			raw_pointer_cast(groupQuerySet_candidates_selected.data()), //output: result are stored here
			raw_pointer_cast(groupQuerySet_candidates_selected_size.data())
			);
}


bool TSGPUManager::dev_check_groupQuery_finished(){
	//note: block number is the number of groups
	check_groupQuery_allItemsFinished<<<this->getGroupQueryNum(),THREAD_PER_BLK,this->getQueryItemsNumberPerGroup()*sizeof(bool)>>>(
			raw_pointer_cast(this->d_groupQuery_info_set.data()),//input:group query
			raw_pointer_cast(this->depressed_d_groupQuery_unfinished.data())//output: the status of every group queries
			);

	bool unfinished = thrust::reduce(depressed_d_groupQuery_unfinished.begin(),depressed_d_groupQuery_unfinished.end());
	return !(unfinished);
}

/**
 * TODO:
 *     add new top-k items into the previous top-k query result, and at the same time update the top-k threshold
 */
void TSGPUManager::dev_maintain_topkCandidates_mergeSort(
		int topk,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,
		device_vector<int>& d_groupQuerySet_candidates_size){


		maintain_topkCandidates_mergeSort<<<this->getTotalQueryItemNum(),THREAD_PER_BLK, 4*topk*sizeof(CandidateEntry)>>>(
		topk,
		raw_pointer_cast(this->d_groupQuerySet_topkResults.data()),
		raw_pointer_cast(d_groupQuerySet_candidates.data()),//input: result are stored here
		raw_pointer_cast(d_groupQuerySet_candidates_size.data()), //note this is size, not address, each query occupied k items size, but some be padded with NULL entry
		raw_pointer_cast(this->d_groupQuerySet_items_lowerBoundThreshold.data())//output: update the threshold for every query items
		);


}


//struct GetCandidateEntryDist{
//
//	__host__ __device__ float op()(CandidateEntry ce){
//		return ce.dist;
//	}
//};

struct GetCandidateEntryDist : public thrust::unary_function<CandidateEntry,float>
  {

	__host__ __device__ GetCandidateEntryDist(){

	}

    __host__ __device__
    float operator()(CandidateEntry& x) const
    {
    	return x.dist;
    }
  };


void TSGPUManager::set_threshold_from_topkResults_byPermutation(int topk){


	host_vector<int> h_pm(this->getTotalQueryItemNum(),0);
	for (int i = 0; i < this->getTotalQueryItemNum(); i++) {
		h_pm[i]=(i+1) * topk  - 1;
	}

	device_vector<int> d_pm = h_pm;

	device_vector<CandidateEntry> d_groupQuerySet_thresholdCandidate(this->getTotalQueryItemNum());

	thrust::copy(thrust::make_permutation_iterator(d_groupQuerySet_topkResults.begin(),
					d_pm.begin()),
					thrust::make_permutation_iterator(d_groupQuerySet_topkResults.begin(),
					d_pm.end()),
					d_groupQuerySet_thresholdCandidate.begin());

	//thrust::transform(groupQuerySet_thresholdCandidate.begin(),groupQuerySet_thresholdCandidate.end(),d_groupQuerySet_items_lowerBoundThreshold,GetCandidateEntryDist());

//	thrust::copy(thrust::make_transform_iterator(d_groupQuerySet_thresholdCandidate.begin(),GetCandidateEntryDist()),
//			thrust::make_transform_iterator(d_groupQuerySet_thresholdCandidate.end(),GetCandidateEntryDist()),
//			d_groupQuerySet_items_lowerBoundThreshold);

	host_vector<CandidateEntry> h_groupQuerySet_thresholdCandidate = d_groupQuerySet_thresholdCandidate;
	host_vector<float> h_groupQuery_items_lowerBoundThreshold(this->getTotalQueryItemNum());

	for(int i=0;i < this->getTotalQueryItemNum(); i++) {//
		h_groupQuery_items_lowerBoundThreshold[i]=h_groupQuerySet_thresholdCandidate[i].dist;
	}

	thrust::copy(h_groupQuery_items_lowerBoundThreshold.begin(),h_groupQuery_items_lowerBoundThreshold.end(),
					this->d_groupQuerySet_items_lowerBoundThreshold.begin());

}

void TSGPUManager::set_threshold_from_topkResults_byCopy(int topk){

	host_vector<CandidateEntry> h_groupQuerySet_topkResults = d_groupQuerySet_topkResults;

	host_vector<float> h_groupQuery_items_lowerBoundThreshold(this->getTotalQueryItemNum());

		for (int i = 0; i < this->getTotalQueryItemNum(); i++) {

			h_groupQuery_items_lowerBoundThreshold[i] =
					h_groupQuerySet_topkResults[(i+1) * topk  - 1].dist;

			}

		//this->d_groupQuerySet_items_lowerBoundThreshold = h_groupQuery_items_lowerBoundThreshold;
		thrust::copy(h_groupQuery_items_lowerBoundThreshold.begin(),h_groupQuery_items_lowerBoundThreshold.end(),
				this->d_groupQuerySet_items_lowerBoundThreshold.begin());
}

/**
 * TODO:
 *     Random select candidates_num ( > topk) and compute the DTW to set a lower bound threshold for topk query
 */
void TSGPUManager::depressed_init_topkThreshold_randomSelect(int topk) {


	//improve here !!! may do a experiment to determine where select more candidates to get a better bound
	int candidates_num = (topk / THREAD_PER_BLK + (topk % THREAD_PER_BLK != 0))	* THREAD_PER_BLK;//to fully utilize the GPU power to a get a tight bounder

	candidates_num =THREAD_PER_BLK;//set candidates num

	depressed_dev_randomSelect_verify_candidates(
			topk,
			candidates_num,
			this->d_groupQuerySet_topkResults);

	set_threshold_from_topkResults_byCopy(topk);

	//print_d_groupQuerySet_items_lowerBoundThreshold();



}

/**
 * TODO:
 *    select verify_candidates_num from *_scan, and store the data into *_select, and finaly select topk candidates into *_topk
 *
 *  note:
 *  1. data flow:
 *  	d_groupQuerySet_candidates_scan => d_groupQuerySet_candidates_selected => d_groupQuerySet_candidates_topk => this->d_groupQuerySet_topkResults
 *  	d_groupQuerySet_candidates_scan is flexible size for each query item, *_selected and *_topk is fixed size for each query item
 */
void TSGPUManager::depressed_dev_randomSelect_and_Verify_topkCandidates(
		int topk,
		int verify_candidates_num,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
		device_vector<int>& d_groupQuerySet_candidates_scan_endIdx,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_topk,
		device_vector<int>& d_groupQuerySet_candidates_topk_size
		){

	device_vector<CandidateEntry> d_groupQuerySet_candidates_selected;
	device_vector<int> d_groupQuerySet_candidates_selected_size;

	//improve here !!! note: this function may be replaced by fast selection algorithm
	//to select the with candidate entries with smaller lower bound
	dev_select_candidates_random(
			verify_candidates_num,
			d_groupQuerySet_candidates_scan,
			d_groupQuerySet_candidates_scan_endIdx,
			d_groupQuerySet_candidates_selected,
			d_groupQuerySet_candidates_selected_size);


	//calculate the true distance between the query items and the selected candidates
	depressed_dev_scan_verifyCandidates_fixedNum(
			verify_candidates_num,
			d_groupQuerySet_candidates_selected,
			d_groupQuerySet_candidates_selected_size);


	//improve here!!! this can be replaced by fast selection,
	//the order of entries in d_groupQuerySet_candidates_topk is not important
	dev_select_candidates_mergeSort_fixedNum(
			topk,
			verify_candidates_num,
			d_groupQuerySet_candidates_selected,
			d_groupQuerySet_candidates_selected_size,
			d_groupQuerySet_candidates_topk,
			d_groupQuerySet_candidates_topk_size);
}

void TSGPUManager::depressed_exact_topk_query_DTW_randomSelect_bucketUnit(int topk){

	depressed_exact_TopK_query_DTW_randomSelect(topk,true);
}

void TSGPUManager::depressed_exact_TopK_query_DTW_randomSelect_bucketWidth(int topk){

	depressed_exact_TopK_query_DTW_randomSelect(topk,false);
}


//====================end the code with random selection

//====================the code with fast k selection



void TSGPUManager::exact_topkQuery_DTW_computeLowerBound(){


	if(enhancedLowerBound){
		this->dev_compute_windowQuery_enhancedLowerBound();
	}else{
		this->dev_compute_windowQuery_lowerBound();

	}

	if(enhancedLowerBound){
		this->dev_compute_groupQuery_enhancedLowerBound();
	}else{

		this->dev_compute_groupQuery_lowerBound();

	}


}




/**
 * TODO:
 * to compute the topk candidates after knowing the lowerBound for each group query and the threshold for topk query
 */
void TSGPUManager::exact_topkQuery_DTW_afterBoundAndThreshold(int topk){

	// data flow:
	//d_groupQuerySet_candidates_scan => d_groupQuerySet_candidates_selected => d_groupQuerySet_candidates_topk => this->d_groupQuerySet_topkResults
	//d_groupQuerySet_candidates_scan is flexible size for each query item, *_selected and *_topk is fixed size for each query item
	device_vector<CandidateEntry> d_groupQuerySet_candidates_scan;
	device_vector<int> d_groupQuerySet_candidates_scan_size;
	device_vector<int> d_groupQuerySet_thread_threshold_endIdx;

	this->profile_cudaTime_start("dev_output_groupQuery_candidates()");
	dev_output_groupQuery_candidates(d_groupQuerySet_candidates_scan,
			d_groupQuerySet_candidates_scan_size,
			d_groupQuerySet_thread_threshold_endIdx
			);//
	this->profile_cudaTime_end("dev_output_groupQuery_candidates()");

	//for exp
	//cout<<"the threshold for filter candidates"<<endl;
	//this->print_d_groupQuerySet_items_lowerBoundThreshold();
	//cout<<"after dev_output_groupQuery_candidates() print un-filtered id"<<endl;
	//print_d_groupQuerySet_candidates_scan_size(d_groupQuerySet_candidates_scan_size);
	//end for exp



	device_vector<int> d_groupQuerySet_candidates_scan_startIdx(d_groupQuerySet_candidates_scan_size.size(),0);
	device_vector<int> d_groupQuerySet_candidates_scan_endIdx(d_groupQuerySet_candidates_scan_size.size(),0);

	thrust::exclusive_scan(d_groupQuerySet_candidates_scan_size.begin(),d_groupQuerySet_candidates_scan_size.end(),d_groupQuerySet_candidates_scan_startIdx.begin());
	thrust::inclusive_scan(d_groupQuerySet_candidates_scan_size.begin(),d_groupQuerySet_candidates_scan_size.end(),d_groupQuerySet_candidates_scan_endIdx.begin());

	//with debug purpose
	//printf("max feature id:%d",this->getMaxFeatureNumber());
	//cout<<"with debug purpose: exact_topkQuery_DTW_afterBoundAndThreshold():after dev_output_groupQuery_candidates()"<<endl;
	//this->print_d_groupQuerySet_candidates_scan_byEndIdx(d_groupQuerySet_candidates_scan,d_groupQuerySet_candidates_scan_endIdx);
	//end with debug purpose

	//for porfiling the unfiltred candidates
	string holder = "A sum:exact_topkQuery_DTW_contNextStep()";
	//string key = prefix_cudatime;
	if(this->exec_cudaEvent_t_set.find(prefix_cudatime+holder)!=exec_cudaEvent_t_set.end()&&enable_sum_unfiltered_candidates){
		this->sum_unfiltered_candidates+=d_groupQuerySet_candidates_scan_endIdx.back();

	}


	this->profile_cudaTime_start("dev_scan_verifyCandidate_perThreadGenerated()");
//	dev_scan_verifyCandidate(
//			d_groupQuerySet_candidates_scan,//input and output: caculate the distance
//			d_groupQuerySet_candidates_scan_endIdx);

	dev_scan_verifyCandidate_perThreadGenerated(
			d_groupQuerySet_candidates_scan,//input and output: caculate the distance
			d_groupQuerySet_thread_threshold_endIdx
			);

	this->profile_cudaTime_end("dev_scan_verifyCandidate_perThreadGenerated()");

	//with debug purpose
	//cout<<"with debug purpose: exact_topkQuery_DTW_afterBoundAndThreshold():d_groupQuerySet_candidates_scan"<<endl;
	//this->print_d_groupQuerySet_candidates_scan_byEndIdx(d_groupQuerySet_candidates_scan,d_groupQuerySet_candidates_scan_endIdx);
	//end with debug purpose


	device_vector<CandidateEntry> groupQuerySet_candidates_selected;
	device_vector<int> groupQuerySet_candidates_selected_size(this->getTotalQueryItemNum(),0);//
	this->profile_clockTime_start("dev_select_candidates_fast()");
	this->profile_cudaTime_start("dev_select_candidates_fast()");
	dev_select_candidates_fast(topk,
				d_groupQuerySet_candidates_scan,
				d_groupQuerySet_candidates_scan_startIdx,
				d_groupQuerySet_candidates_scan_endIdx,
				groupQuerySet_candidates_selected //output
				);
	this->profile_cudaTime_end("dev_select_candidates_fast()");
	this->profile_clockTime_end("dev_select_candidates_fast()");



	dev_capIntVector(topk,d_groupQuerySet_candidates_scan_size,groupQuerySet_candidates_selected_size);

	//with debug purpose///
	//cout<<"with debug purpose: exact_topkQuery_DTW_afterBoundAndThreshold():groupQuerySet_candidates_selected"<<endl;
	//this->print_d_groupQuerySet_candidates_verified(topk,groupQuerySet_candidates_selected,groupQuerySet_candidates_selected_size);
	//end with debug purpose

	//with debug purpose
	//cout<<"with debug purpose:exact_topkQuery_DTW_afterBoundAndThreshold() before mergeSort"<<endl;
	//this->print_d_groupQuerySet_topkResults(topk);
	//end with debug purpose


	this->dev_select_candidates_mergeSort_fixedNum(
				topk,
				topk,
				groupQuerySet_candidates_selected,
				groupQuerySet_candidates_selected_size,
				this->d_groupQuerySet_topkResults,
				this->d_groupQuerySet_topkResults_size);

	//with debug purpose
	//cout<<"with debug purpose:exact_topkQuery_DTW_afterBoundAndThreshold() after mergeSort"<<endl;
	//this->print_d_groupQuerySet_topkResults(topk);
	//end with debug purpose

}

void TSGPUManager::exact_topkQuery_DTW(int topk){
	exact_topkQuery_DTW_computeLowerBound();

	init_topkThreshold_fastSelect(topk);

	exact_topkQuery_DTW_afterBoundAndThreshold(topk);


	//cout<<"print threshold after dev_maintain_topkCandidates_mergeSort()"<<endl;
	//this->print_d_groupQuerySet_items_lowerBoundThreshold();

}

void TSGPUManager::exact_topkQuery_DTW_contFirstStep(int topk){
	exact_topkQuery_DTW(topk);
}

/**
 * TODO:
 * in the continous query, estimate the topk-threshold from the topk-result of previous step
 */
void TSGPUManager::estimate_topkThreshold_contNextStep(int topk){//


		device_vector<CandidateEntry> d_groupQuerySet_candidates = this->d_groupQuerySet_topkResults;
		device_vector<int> d_groupQuerySet_candidates_size=this->d_groupQuerySet_topkResults_size;


		this->dev_scan_verifyCandidates_fixedNum(//
	  			topk,
	  			d_groupQuerySet_candidates,//input and output: calculate the distance
	  			d_groupQuerySet_candidates_size);


		this->dev_select_candidates_mergeSort_fixedNum(
				topk,
				topk,
				d_groupQuerySet_candidates,
				d_groupQuerySet_candidates_size,
				this->d_groupQuerySet_topkResults,
				this->d_groupQuerySet_topkResults_size);


		set_threshold_from_topkResults_byCopy(topk);

}


void TSGPUManager::exact_topkQuery_DTW_contNextStep_withInitThreshold(int topk){

	this->profile_cudaTime_start("exact_topkQuery_DTW_computeLowerBound()");
	exact_topkQuery_DTW_computeLowerBound();
	this->profile_cudaTime_end("exact_topkQuery_DTW_computeLowerBound()");

	this->profile_cudaTime_start("init_topkThreshold_fastSelect()");
	init_topkThreshold_fastSelect(topk);
	this->profile_cudaTime_end("init_topkThreshold_fastSelect()");

	this->profile_cudaTime_start("exact_topkQuery_DTW_afterBoundAndThreshold()");
	exact_topkQuery_DTW_afterBoundAndThreshold(topk);
	this->profile_cudaTime_end("exact_topkQuery_DTW_afterBoundAndThreshold()");
}

void TSGPUManager::exact_topkQuery_DTW_contNextStep(int topk){


	this->profile_cudaTime_start("exact_topkQuery_DTW_computeLowerBound()");
	exact_topkQuery_DTW_computeLowerBound();
	this->profile_cudaTime_end("exact_topkQuery_DTW_computeLowerBound()");

	this->profile_cudaTime_start("estimate_topkThreshold_contNextStep()");
	estimate_topkThreshold_contNextStep(topk);
	this->profile_cudaTime_end("estimate_topkThreshold_contNextStep()");

	this->profile_cudaTime_start("exact_topkQuery_DTW_afterBoundAndThreshold()");
	exact_topkQuery_DTW_afterBoundAndThreshold(topk);
	this->profile_cudaTime_end("exact_topkQuery_DTW_afterBoundAndThreshold()");


}


void TSGPUManager::depressed_exact_TopK_query_DTW_fastSelect_bucketUnit(int topk){

	depressed_exact_TopK_query_DTW_fastSelect(topk,true);
}

void TSGPUManager::depressed_exact_TopK_query_DTW_fastSelect_bucketWidth(int topk){

	depressed_exact_TopK_query_DTW_fastSelect(topk,false);
}





/**
 *
 */
void TSGPUManager::depressed_exact_TopK_query_DTW_fastSelect(int topk, bool isBucketUnit ){




	bool finished = false;

	int iteration = 0;

	//first step, get the bound for topk
	if (isBucketUnit) {
		depressed_dev_BidirectionExpansion_bucketUnit();
		depressed_dev_compute_windowQuery_LowerBound_bucketUnit();


	} else {
		depressed_dev_BidirectionExpansion_bucket_exclusive();
		depressed_dev_compute_windowQuery_LowerBound_bucket_inclusive();
	}

	depressed_dev_compute_groupQuery_LowerBound();


	init_topkThreshold_fastSelect(topk);



	// data flow:
    //d_groupQuerySet_candidates_scan => d_groupQuerySet_candidates_selected => d_groupQuerySet_candidates_topk => this->d_groupQuerySet_topkResults
	//d_groupQuerySet_candidates_scan is flexible size for each query item, *_selected and *_topk is fixed size for each query item
	device_vector<CandidateEntry> d_groupQuerySet_candidates_scan;
	device_vector<int> d_groupQuerySet_candidates_scan_size;

	device_vector<CandidateEntry> d_groupQuerySet_candidates_verified;
	device_vector<int> d_groupQuerySet_candidates_verified_size;

	while(!finished){
		cout<<"topk-query iteration:"<<iteration<<endl;
		iteration++;

		if(isBucketUnit){
			depressed_dev_BidirectionExpansion_bucketUnit();
			depressed_dev_compute_windowQuery_LowerBound_bucketUnit();
		}else{
			depressed_dev_BidirectionExpansion_bucket_exclusive();
			depressed_dev_compute_windowQuery_LowerBound_bucket_inclusive();
		}

		depressed_dev_compute_groupQuery_LowerBound();

		depressed_dev_output_groupQuery_candidates(d_groupQuerySet_candidates_scan,d_groupQuerySet_candidates_scan_size);//


		finished = dev_check_groupQuery_finished();
		cout<<" finished ="<<finished<<" with iteration:"<<iteration<<endl;
		if(finished||iteration>5) break;


		depressed_dev_fastSelect_and_verify_topkCandidates(
				topk,
				d_groupQuerySet_candidates_scan,
				d_groupQuerySet_candidates_scan_size,
				d_groupQuerySet_candidates_verified,
				d_groupQuerySet_candidates_verified_size);



		dev_capIntVector(topk,d_groupQuerySet_candidates_scan_size,d_groupQuerySet_candidates_verified_size);

		dev_maintain_topkCandidates_mergeSort(
					topk,
					d_groupQuerySet_candidates_verified,
					d_groupQuerySet_candidates_verified_size);//will be changed back to d_groupQuerySet_candidates_verified_size
		cout<<"print threshold after dev_maintain_topkCandidates_mergeSort()"<<endl;
		this->print_d_groupQuerySet_items_lowerBoundThreshold();
	}


}

struct compare_groupQueryLowerBound
  {
    __host__ __device__
    bool operator()(float lhs, float rhs)
    {
      if((lhs<=INITIAL_LABEL_VALUE+0.5)&&(lhs>=INITIAL_LABEL_VALUE-0.5))	return true;

      return lhs < rhs;
    }
  };


void TSGPUManager::getMinMax_d_groupQuerySet_lowerBound(float& min, float& max){

		min=0;
		device_vector<float>::iterator maxDataItr =  thrust::max_element(d_groupQuerySet_lowerBound.begin(), d_groupQuerySet_lowerBound.end(),compare_groupQueryLowerBound());
		device_vector<float> d_max(1);
		thrust::copy(maxDataItr, maxDataItr + 1, d_max.begin());
		host_vector<float> h_max;
		h_max = d_max;


		max = h_max[0];

}


/**
 * TODO:
 *     verify the candidate with smallest groupQuery_lowerBound to get initial threshold for topk
 */
void TSGPUManager::init_topkThreshold_fastSelect(int topk){


	int candidates_num = topk;
	float min=0,max;
	getMinMax_d_groupQuerySet_lowerBound( min,  max);

	device_vector<int> d_groupQuerySet_lowerBound_size(this->getTotalQueryItemNum(),this->getMaxFeatureNumber());
	device_vector<int> d_groupQuerySet_lowerBound_startIdx(this->getTotalQueryItemNum(), 0);
	device_vector<int> d_groupQuerySet_lowerBound_endIdx(this->getTotalQueryItemNum(), 0);

	thrust::exclusive_scan(d_groupQuerySet_lowerBound_size.begin(),d_groupQuerySet_lowerBound_size.end(),d_groupQuerySet_lowerBound_startIdx.begin());
	thrust::inclusive_scan(d_groupQuerySet_lowerBound_size.begin(),d_groupQuerySet_lowerBound_size.end(),d_groupQuerySet_lowerBound_endIdx.begin());


	host_vector<int> h_groupQuerySet_lowerBound_startIdx = d_groupQuerySet_lowerBound_startIdx;

  	device_vector<int> d_groupQuerySet_fastSel_candidates_size(this->getTotalQueryItemNum(),candidates_num);
  	device_vector<int> d_groupQuerySet_fastSel_candidates_idSet;

  	device_vector<float> min_vec(this->getTotalQueryItemNum(),min);
  	device_vector<float> max_vec(this->getTotalQueryItemNum(),max);



  	tailored_bucket_topk(&this->d_groupQuerySet_lowerBound,//
  			ValueOfGroupQuery_boundEntry(), &min_vec,&max_vec,
  			candidates_num,
  			&d_groupQuerySet_lowerBound_startIdx, &d_groupQuerySet_lowerBound_endIdx,
  			this->getTotalQueryItemNum(),
  			&d_groupQuerySet_fastSel_candidates_idSet);


  	host_vector<int> h_groupQuerySet_fastSel_candidates_idSet = d_groupQuerySet_fastSel_candidates_idSet;
  	host_vector<CandidateEntry> h_groupQuerySet_candidates(h_groupQuerySet_fastSel_candidates_idSet.size());

  	//with debug purpose
  	//this->print_d_groupQuerySet_lowerBound_selected_idSet(candidates_num,d_groupQuerySet_fastSel_candidates_idSet,
  	//		d_groupQuerySet_lowerBound,d_groupQuerySet_lowerBound_endIdx);

  	//end with debug purpose


  	for(int i=0;i<h_groupQuerySet_candidates.size();i++){
  	  	h_groupQuerySet_candidates[i].feature_id = (h_groupQuerySet_fastSel_candidates_idSet[i] - h_groupQuerySet_lowerBound_startIdx[i/candidates_num]) ;
  	 }
  	device_vector<CandidateEntry> d_groupQuerySet_candidates = h_groupQuerySet_candidates;

	host_vector<CandidateEntry> h_gc = d_groupQuerySet_candidates;

  	this->dev_scan_verifyCandidates_fixedNum(
  			candidates_num,
  			d_groupQuerySet_candidates,//input and output: caculate the distance
  			d_groupQuerySet_fastSel_candidates_size);

  	//with debug purpose
  	//cout<<"ith debug purpose: print d_groupQuerySet_candidates"<<endl;
  	//this->print_d_groupQuerySet_candidates_verified(topk,d_groupQuerySet_candidates,d_groupQuerySet_fastSel_candidates_size);
  	//end with debug purpose

	this->dev_select_candidates_mergeSort_fixedNum(
			topk,
			candidates_num,
			d_groupQuerySet_candidates,
			d_groupQuerySet_fastSel_candidates_size,
			this->d_groupQuerySet_topkResults,
			this->d_groupQuerySet_topkResults_size);

	//with debug purpose
	//cout<<"with debug purpose: before mergesort, print d_groupQuerySet_topkResults"<<endl;
	//this->print_d_groupQuerySet_topkResults(topk);
	//end with debug purpose

	set_threshold_from_topkResults_byCopy(topk);


}




void TSGPUManager::getMinMax_groupQuerySet_candidates(device_vector<CandidateEntry>& d_groupQuerySet_candidates, CandidateEntry& min, CandidateEntry& max){

		min.dist = 0;
		//device_vector<CandidateEntry>::iterator minDataItr =  thrust::min_element(d_groupQuerySet_candidates.begin(), d_groupQuerySet_candidates.end(),compare_CandidateEntry_dist());
		device_vector<CandidateEntry>::iterator maxDataItr =  thrust::max_element(d_groupQuerySet_candidates.begin(), d_groupQuerySet_candidates.end(),compare_CandidateEntry_dist());

		//device_vector<CandidateEntry> d_min(1),d_max(1);
		device_vector<CandidateEntry> d_max(1);
		//thrust::copy(minDataItr, minDataItr + 1, d_min.begin());
		thrust::copy(maxDataItr, maxDataItr + 1, d_max.begin());
		//host_vector<CandidateEntry> h_min,h_max;
		host_vector<CandidateEntry> h_max;
		//h_min = d_min;
		h_max = d_max;
		//min = h_min[0];
		max = h_max[0];

}




void TSGPUManager::depressed_dev_select_candidates_fast(int selectedCandidates_num,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,
		device_vector<int>& d_groupQuerySet_candidates_startIdx,
		device_vector<int>& d_groupQuerySet_candidates_endIdx,
		device_vector<CandidateEntry>& groupQuerySet_candidates_selected //output
		) {

		//CandidateEntry ce_min, ce_max;
		//getMinMax_groupQuerySet_candidates(d_groupQuerySet_candidates, ce_min,  ce_max);

		//float min = ce_min.dist;
		//float max = ce_max.dist;

		device_vector<float> min_vec(this->getTotalQueryItemNum(),0);
		device_vector<float> max_vec(this->getTotalQueryItemNum(),0);

		thrust::copy(this->d_groupQuerySet_items_lowerBoundThreshold.begin(),d_groupQuerySet_items_lowerBoundThreshold.end(),max_vec.begin());


		//device_vector<int> d_groupQuerySet_candidates_selectedNum(this->getTotalQueryItemNum(),selectedCandidates_num);
		device_vector<int> d_groupQuerySet_candidates_selected_idSet;


		tailored_bucket_topk(&d_groupQuerySet_candidates, ValueOfCandidateEntry(), &min_vec, &max_vec,
				selectedCandidates_num,
			&d_groupQuerySet_candidates_startIdx,
			&d_groupQuerySet_candidates_endIdx,
			this->getTotalQueryItemNum(),
			&d_groupQuerySet_candidates_selected_idSet);




		//every query item has at most "selectedCandidates_num" items,
		//if candidates is less than that, the following part is padded with empty
		groupQuerySet_candidates_selected.clear();
		groupQuerySet_candidates_selected.resize(selectedCandidates_num*this->getTotalQueryItemNum());
		groupQuerySet_candidates_selected.shrink_to_fit();

		thrust::copy(thrust::make_permutation_iterator(d_groupQuerySet_candidates.begin(),d_groupQuerySet_candidates_selected_idSet.begin()),
				thrust::make_permutation_iterator(d_groupQuerySet_candidates.begin(),d_groupQuerySet_candidates_selected_idSet.end()),
				groupQuerySet_candidates_selected.begin()
		);
}


struct IncreaseOperator{


__host__ __device__ IncreaseOperator(){

}


__host__ __device__ float operator()(float data)
{
   		return data*1.01;

}
};

//for fix the bugs of bucket-k-selection for function dev_select_candidates_fast(), just increase the max_vec by 1.01
void dev_IncreaseOperator(device_vector<float>& org, device_vector<float>& _trans){

	thrust::transform(org.begin(), org.end(), _trans.begin(),
			IncreaseOperator());

}

void TSGPUManager::dev_select_candidates_fast(int selectedCandidates_num,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,
		device_vector<int>& d_groupQuerySet_candidates_startIdx,
		device_vector<int>& d_groupQuerySet_candidates_endIdx,
		device_vector<CandidateEntry>& groupQuerySet_candidates_selected //output
		) {


		//CandidateEntry min;
		//CandidateEntry max;
		//getMinMax_groupQuerySet_candidates(d_groupQuerySet_candidates, min, max);



		//device_vector<int> d_groupQuerySet_candidates_selectedNum(
		//	this->getTotalQueryItemNum(), selectedCandidates_num);
		device_vector<int> d_groupQuerySet_candidates_selected_idSet;
		device_vector<float> min_vec(d_groupQuerySet_items_lowerBoundThreshold.size(),0);
		device_vector<float> max_vec=d_groupQuerySet_items_lowerBoundThreshold;


		//for fix the bugs of bucket-k-selection for function dev_select_candidates_fast(), just increase the max_vec by 1.01
		dev_IncreaseOperator(max_vec, max_vec);

		this->profile_cudaTime_start("tailored_bucket_topk()");
		this->profile_clockTime_start("tailored_bucket_topk()");
		tailored_bucket_topk(&d_groupQuerySet_candidates, ValueOfCandidateEntry(),
			&min_vec,&max_vec,
			//min.dist,max.dist,
			selectedCandidates_num,
			&d_groupQuerySet_candidates_startIdx,
			&d_groupQuerySet_candidates_endIdx, this->getTotalQueryItemNum(),
			&d_groupQuerySet_candidates_selected_idSet);
		this->profile_clockTime_end("tailored_bucket_topk()");
		this->profile_cudaTime_end("tailored_bucket_topk()");



		//with debug purpose//
		//cout<<"with debug purpose: dev_select_candidates_fast():d_groupQuerySet_candidates_selected_idSet "<<endl;
		//this->print_d_groupQuerySet_candidates_selected_idSet(selectedCandidates_num,d_groupQuerySet_candidates_selected_idSet,d_groupQuerySet_candidates,d_groupQuerySet_candidates_endIdx);
		//end with debug purpose

		//every query item has at most "selectedCandidates_num" items,
		//if candidates is less than that, the following part is padded with empty
		groupQuerySet_candidates_selected.clear();
		groupQuerySet_candidates_selected.shrink_to_fit();
		groupQuerySet_candidates_selected.resize(selectedCandidates_num * this->getTotalQueryItemNum());

		//with debug purpose
		//this->print_d_groupQuerySet_candidates_scan_byEndIdx(d_groupQuerySet_candidates,d_groupQuerySet_candidates_endIdx);

		thrust::copy(
				thrust::make_permutation_iterator(d_groupQuerySet_candidates.begin(),
						d_groupQuerySet_candidates_selected_idSet.begin()),
				thrust::make_permutation_iterator(d_groupQuerySet_candidates.end(),
						d_groupQuerySet_candidates_selected_idSet.end()),
				groupQuerySet_candidates_selected.begin());

		//with debug purpose
		//cout<<"with debug purpose: dev_select_candidates_fast():d_groupQuerySet_candidates_selected_idSet "<<endl;
		//with debug purpose
		//this->print(selectedCandidates_num,d_groupQuerySet_candidates_selected_idSet,d_groupQuerySet_candidates,d_groupQuerySet_candidates_endIdx);
		//end with debug purpose

}

struct CapOperator{
	int cap;

__host__ __device__ CapOperator(int cap){
	this->cap = cap;
}


__host__ __device__ int operator()(int data)
{
   		return data<cap? data:cap;
}
};

void TSGPUManager::dev_capIntVector(int cap,device_vector<int>& org, device_vector<int>& _trans){

	thrust::transform(org.begin(), org.end(), _trans.begin(),
			CapOperator(cap));

}

void TSGPUManager::depressed_dev_fastSelect_and_verify_topkCandidates(int topk,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
		device_vector<int>& d_groupQuerySet_candidates_scan_size,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_topk,
		device_vector<int>& d_groupQuerySet_candidates_topk_size) {


	device_vector<int> d_groupQuerySet_candidates_scan_startIdx(d_groupQuerySet_candidates_scan_size.size(),0);
	device_vector<int> d_groupQuerySet_candidates_scan_endIdx(d_groupQuerySet_candidates_scan_size.size(),0);

	thrust::exclusive_scan(d_groupQuerySet_candidates_scan_size.begin(),d_groupQuerySet_candidates_scan_size.end(),d_groupQuerySet_candidates_scan_startIdx.begin());
	thrust::inclusive_scan(d_groupQuerySet_candidates_scan_size.begin(),d_groupQuerySet_candidates_scan_size.end(),d_groupQuerySet_candidates_scan_endIdx.begin());


	depressed_dev_select_candidates_fast(topk,
			d_groupQuerySet_candidates_scan,
			d_groupQuerySet_candidates_scan_startIdx,
			d_groupQuerySet_candidates_scan_endIdx,
			d_groupQuerySet_candidates_topk //output
			);

	d_groupQuerySet_candidates_topk_size.clear();
	d_groupQuerySet_candidates_topk_size.resize(this->getTotalQueryItemNum(),0);//imporve here !!! this should be set by bucket_topk
	d_groupQuerySet_candidates_topk_size.shrink_to_fit();

	dev_capIntVector(topk,d_groupQuerySet_candidates_scan_size,d_groupQuerySet_candidates_topk_size);

	depressed_dev_scan_verifyCandidates_fixedNum(
			topk,
			d_groupQuerySet_candidates_topk,
			d_groupQuerySet_candidates_topk_size);
}



//==================end the code with fast selection

int TSGPUManager::getMaxDimensionsOfGroupQueries(vector<GroupQuery_info*>& gpuGroupQuery_info_set){
	int max = -1;
	for(int i=0;i<gpuGroupQuery_info_set.size();i++){

		max = gpuGroupQuery_info_set[i]->getMaxQueryLen()<=max ? max:gpuGroupQuery_info_set[i]->getMaxQueryLen();

	}
	return max;
}

int TSGPUManager::depressed_getMaxWindowFeatureNumber(){
	return depressed_gpuManager.getMaxFeatureNumber();
}

int TSGPUManager::comp_maxWindowFeatureNumber(){

	host_vector<int>::iterator maxLenItr = thrust::max_element(this->h_ts_blade_len.begin(),this->h_ts_blade_len.end());
	host_vector<int> h_maxLen(1);
	thrust::copy(maxLenItr,maxLenItr+1,h_maxLen.begin());
	return h_maxLen[0]/this->winDim;

}

void TSGPUManager::initMemeberVariables(){
		this->winDim = 0;
		this->groupQuery_maxdimension = 0;
		this->d_groupQuery_info_set.clear();
		this->d_windowQuery_lowerBound.clear();
		this->d_windowQuery_lowerBound_q2d.clear();
		this->d_groupQuerySet_lowerBound.clear();

		this->d_ts_data_blade_endIdx.clear();
		this->d_ts_data.clear();
	}


void TSGPUManager::depressed_update_GroupQueryInfo_vec_fromHostToDevice( host_vector<GroupQuery_info*> hvec, device_vector<GroupQuery_info*> dvec ){

	for(int i=0;i<hvec.size();i++){
		depressed_update_GroupQueryInfo_fromHostToDevice(hvec[i],dvec[i]);
	}
}

void TSGPUManager::update_GroupQueryInfo_vec_dataAndStartWindId_fromHostToDevice( host_vector<GroupQuery_info*> hvec, device_vector<GroupQuery_info*> dvec ){

	for(int i=0;i<hvec.size();i++){
		update_GroupQueryInfo_dataAndStartWindId_fromHostToDevice(hvec[i],dvec[i]);
	}
}


void TSGPUManager::depressed_update_ContQueryInfo(int groupQueryInfo_id,vector<float> gqi_data, int gqi_data_newData_indicator){

	assert(gqi_data.size() == this->groupQuery_maxdimension);

	//update data array firstly
	GroupQuery_info* h_gqi = h_groupQuery_info_set[groupQueryInfo_id];
	for(int i=0;i<this->groupQuery_maxdimension;i++){
		h_gqi->data[i] = gqi_data[i];
	}



	//create new sliding windows id
	int new_window_start = ((gqi_data_newData_indicator-winDim+1)>=0)? (gqi_data_newData_indicator-winDim+1):0;
	int slidingWindowNum = this->groupQuery_maxdimension-winDim+1;
	increase_startwindowId(h_gqi,slidingWindowNum-new_window_start);

	vector<WindowQueryInfo*> qi_vec;
	qi_vec.reserve(slidingWindowNum);
	//add new window query id
	for(int i=new_window_start;i<slidingWindowNum;i++){

		int d_windowQuery_id = getWinQueryId(h_gqi,i) + groupQueryInfo_id*this->windowNum_perGroup;

		GpuWindowQuery gpuQuery_sw(d_windowQuery_id, this->h_groupQuery_info_set[groupQueryInfo_id]->blade_id,winDim);

		this->depressed_set_windowQueryInfo_GpuQuery_newEntry(groupQueryInfo_id,i,gpuQuery_sw);

		// use gpuQuery_sw to update gpuMananger
		depressed_gpuManager.update_windowQuery_entryAndTable(gpuQuery_sw);

	}

	if(!(this->depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled)){
		depressed_update_winQuery_LBKeogh_ACDUpdate(new_window_start, groupQueryInfo_id);
	}

	h_gqi->depressed_reset_item_query_asUnfinished();
}


/**
 * gqi_data_set: data area part to update groupquery info
 * gqi_data_sliding_indicator: it labels the starting position of "new" data compared with previous existing data (after and inclusive this postion, all data are NEW)
 */
void TSGPUManager::update_ContQueryInfo(int groupQueryInfo_id,vector<float> gqi_data, int gqi_data_newData_indicator){

	assert(gqi_data.size() == this->groupQuery_maxdimension);

	//update data array firstly
	GroupQuery_info* h_gqi = h_groupQuery_info_set[groupQueryInfo_id];
	for(int i=0;i<this->groupQuery_maxdimension;i++){
		h_gqi->data[i] = gqi_data[i];
	}



	//create new sliding windows id
	//gqi_data_newData_indicator-winDim+1<=0 means all query data is updated, the starting window id just like fresh query coming
	int new_window_start = ((gqi_data_newData_indicator-winDim+1)>=0)? (gqi_data_newData_indicator-winDim+1):0;
	int slidingWindowNum = this->groupQuery_maxdimension-winDim+1;
	increase_startwindowId(h_gqi,slidingWindowNum-new_window_start);//the window with smaller than startWindowId is pruned

	vector<WindowQueryInfo*> qi_vec;
	qi_vec.reserve(slidingWindowNum);

	//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():forloop");
	//add new window query id
	for(int i=new_window_start;i<slidingWindowNum;i++){

		int d_windowQuery_id = getWinQueryId(h_gqi,i) + groupQueryInfo_id*this->windowNum_perGroup;

		//GpuWindowQuery gpuWindowQuery_sw(d_windowQuery_id, this->h_groupQuery_info_set[groupQueryInfo_id]->blade_id,winDim);//for imp2
		//this->set_windowQueryInfo_GpuQuery_newEntry(groupQueryInfo_id,i,gpuWindowQuery_sw);//for imp2
		// use gpuQuery_sw to update gpuMananger
		//update_windowQuery_entryAndTable(gpuWindowQuery_sw);//for imp2

		update_windowQuery_entryAndTable(d_windowQuery_id,groupQueryInfo_id, i);

	}
	//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():forloop");


	//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable()");
	update_winQuery_LBKeoghAndTable(new_window_start, groupQueryInfo_id);
	//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable()");


	//h_gqi->depressed_reset_item_query_asUnfinished();
}

/**
 * depressed: replaced by labelReset_windowQuery_lowerBound() and dev_batch_reset_windowQuery_lowerBound();
 */
void TSGPUManager::depressed_reset_windowQuery_lowerBound(int windowQuery_info_id){

	int wqi_start = windowQuery_info_id*this->getMaxWindowFeatureNumber();
	int wqi_end = wqi_start+this->getMaxWindowFeatureNumber();

	this->h_windowQuery_LBKeogh_updatedLabel[windowQuery_info_id]=true;
	thrust::fill(this->d_windowQuery_lowerBound.begin()+wqi_start,this->d_windowQuery_lowerBound.begin()+wqi_end, 0);//clear the Count&ACD table of this window query//
	if(enhancedLowerBound){
		thrust::fill(this->d_windowQuery_lowerBound_q2d.begin()+wqi_start,this->d_windowQuery_lowerBound_q2d.begin()+wqi_end, 0);//clear the Count&ACD table of this window query//
	}
}


void TSGPUManager::labelReset_windowQuery_lowerBound(int windowQuery_info_id){
	this->h_windowQuery_LBKeogh_updatedLabel[windowQuery_info_id]=true;
}

void TSGPUManager::update_windowQuery_entryAndTable(WindowQueryInfo* h_windowQueryInfo, int d_windowQuery_info_id){

	update_windowQueryInfo_entry_fromHostToDevice( h_windowQueryInfo, this->d_windowQuery_info_set[d_windowQuery_info_id]);
	delete this->h_windowQuery_info_set[ d_windowQuery_info_id];
	h_windowQuery_info_set[ d_windowQuery_info_id]  = h_windowQueryInfo;

	//reset_windowQuery_lowerBound(d_windowQuery_info_id);
	labelReset_windowQuery_lowerBound(d_windowQuery_info_id);//for reset with batch method


}

void TSGPUManager::update_windowQuery_entryAndTable(GpuWindowQuery& windowQuery){

	WindowQueryInfo* hqi = new WindowQueryInfo(windowQuery);

	update_windowQuery_entryAndTable(hqi, windowQuery.queryId);

	//delete hqi;

}

void TSGPUManager::update_windowQuery_entryAndTable(int winQueryId,int groupQueryId, int slidingWin_dataIdx){//for imp

	GroupQuery_info* groupQuery_info = this->h_groupQuery_info_set[groupQueryId];

	WindowQueryInfo* hqi = new WindowQueryInfo(winDim,groupQuery_info->blade_id);
	hqi->keyword=new float[winDim];
	for (int kj = 0; kj < winDim; kj++) {
		hqi->keyword[kj] = groupQuery_info->data[slidingWin_dataIdx + kj];
	}

	update_windowQuery_entryAndTable(hqi, winQueryId);
}






/**
 * update the LB_keogh for the existing window query before new_windowQuery_id
 *
 *some useful inference but not used in this function:
 * 	the affected window query: [new_window_start-sc_band, new_window_start)
	suppose the affected window query is affectedQueryId =
	the affected dimension for this query is winDim-1 - (sc_band-(new_window_start-affectedQueryId))
 *
 * NOTE: private function, this function is auxiliary function of update_GroupQueryInfo().
 */
void TSGPUManager::depressed_update_winQuery_LBKeogh_ACDUpdate(int new_window_start, int groupQueryInfo_id ){

	GroupQuery_info* h_gqi = h_groupQuery_info_set[groupQueryInfo_id];

	int affectedQueryId_start =
			(new_window_start - sc_band) < 0 ?
					0 : (new_window_start - sc_band);

	for (int i = affectedQueryId_start; i < new_window_start; i++) {

		vector<float> new_upwardDistanceBound(this->winDim, 0),
				new_downwardDistanceBound(this->winDim, 0);
		int windowQuery_id = getWinQueryId(h_gqi, i)
						+ groupQueryInfo_id * this->windowNum_perGroup;
		WindowQueryInfo* h_qi = depressed_gpuManager.getQueryInfo(windowQuery_id);

		initQueryItem_LBKeogh(sc_band, h_gqi->data, i, groupQuery_maxdimension,
				this->winDim, h_qi->keyword,
				new_upwardDistanceBound, new_downwardDistanceBound);


		//get the expansion of lower and upper bound
		int affected_dim_start = winDim - sc_band;
		int updateValueVec_start = windowQuery_id * sc_band;

		for (int j = 0; j < sc_band; j++) {
			int affected_dim = affected_dim_start + j;
			float upExp = new_upwardDistanceBound[affected_dim]
					- h_qi->upperBoundDist[affected_dim];
			float downExp = new_downwardDistanceBound[affected_dim]
					- h_qi->lowerBoundDist[affected_dim];
			assert(upExp >= 0 && downExp >= 0);
			float acdExp = std::max(upExp, downExp);

			this->depressed_h_winQuery_LBKeogh_ACDUpdate_valueVec[updateValueVec_start + j] =
					acdExp;
		}
		depressed_gpuManager.update_QueryInfo_entry_upperAndLowerBound(windowQuery_id,new_upwardDistanceBound,new_downwardDistanceBound);

		this->depressed_h_winQuery_LBKeogh_ACDUpdate_subCtrlVec[windowQuery_id] = true;

	}
}


/**
 * update the LB_keogh for the existing window query before new_windowQuery_id
 * the function is to update to new LBKeogh bound for all affected window queries when new data points are appending to the query
 *
 *some useful inference but not used in this function:
 * 	the affected window query: [new_window_start-sc_band, new_window_start)
	suppose the affected window query is affectedQueryId =
	the affected dimension for this query is winDim-1 - (sc_band-(new_window_start-affectedQueryId))


 *
 * NOTE: 1. private function, this function is auxiliary function of update_GroupQueryInfo().
 *       2. must reset the d_windowQuery_lowerBound because this bound will be re-caculated
 */
void TSGPUManager::update_winQuery_LBKeoghAndTable(int new_window_start, int groupQueryInfo_id ){

	GroupQuery_info* h_gqi = h_groupQuery_info_set[groupQueryInfo_id];

	int affectedQueryId_start =
			(new_window_start - sc_band) < 0 ?
					0 : (new_window_start - sc_band);

	profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop");
	for (int i = affectedQueryId_start; i < new_window_start; i++) {

		//vector<float> new_upwardDistanceBound(this->winDim, 0),//for imp:
		//		new_downwardDistanceBound(this->winDim, 0);
		int windowQuery_id = getWinQueryId(h_gqi, i) + groupQueryInfo_id * this->windowNum_perGroup;

		//WindowQueryInfo* h_qi = h_windowQuery_info_set[windowQuery_id];//depressed_gpuManager.getQueryInfo(windowQuery_id);//for imp:

		//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop:initQueryItem_LBKeogh()");
		//initQueryItem_LBKeogh(sc_band, h_gqi->data, i, groupQuery_maxdimension,//for imp:no calculation for upper and lower bound on host window query
		//		this->winDim, h_qi->keyword,
		//		new_upwardDistanceBound, new_downwardDistanceBound);
		//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop:initQueryItem_LBKeogh()");

		//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop:update_windowQueryInfo_entry_upperAndLowerBound()");
		//update_windowQueryInfo_entry_upperAndLowerBound(windowQuery_id,new_upwardDistanceBound,new_downwardDistanceBound);//for imp:no calculation for upper and lower bound on host window query
		//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop:update_windowQueryInfo_entry_upperAndLowerBound()");

		//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop:reset_windowQuery_lowerBound()");
		//reset_windowQuery_lowerBound(windowQuery_id);
		labelReset_windowQuery_lowerBound(windowQuery_id);//for reset with batch method
		//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop:reset_windowQuery_lowerBound()");
		//this->h_windowQuery_LBKeogh_updatedLabel[windowQuery_id]=true;

	}
	profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo():update_winQuery_LBKeoghAndTable():forloop");
}


void TSGPUManager::update_windowQueryInfo_entry_fromHostToDevice(WindowQueryInfo* host_queryInfo, WindowQueryInfo* device_queryInfo){

		//copy data

		//cudaMemcpy(&(device_queryInfo->depressed_topK), &(host_queryInfo->depressed_topK), sizeof(int), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(device_queryInfo->numOfDimensionToSearch), &(host_queryInfo->numOfDimensionToSearch), sizeof(int), cudaMemcpyHostToDevice);//for imp2
		//cudaMemcpy(&(device_queryInfo->depressed_distFuncType), &(host_queryInfo->depressed_distFuncType), sizeof(int), cudaMemcpyHostToDevice);

		int size = host_queryInfo->numOfDimensionToSearch;
		WindowQueryInfo* d2h_queryInfo = (WindowQueryInfo*) malloc(sizeof(WindowQueryInfo));//for imp2
		cudaMemcpy(d2h_queryInfo, device_queryInfo, sizeof(WindowQueryInfo), cudaMemcpyDeviceToHost);//for imp2

		//cudaMemcpy(d2h_queryInfo->depressed_searchDim, host_queryInfo->depressed_searchDim, sizeof(int) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(d2h_queryInfo->depressed_distanceFunc, host_queryInfo->depressed_distanceFunc, sizeof(float) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(d2h_queryInfo->depressed_upperBoundSearch, host_queryInfo->depressed_upperBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(d2h_queryInfo->depressed_lowerBoundSearch, host_queryInfo->depressed_lowerBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(d2h_queryInfo->depressed_lastPos, host_queryInfo->depressed_lastPos, sizeof(int2) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->keyword, host_queryInfo->keyword, sizeof(float) * size, cudaMemcpyHostToDevice);//for imp2
		//cudaMemcpy(d2h_queryInfo->depressed_dimWeight, host_queryInfo->depressed_dimWeight, sizeof(float) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(d2h_queryInfo->upperBoundDist, host_queryInfo->upperBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);//for imp2
		//cudaMemcpy(d2h_queryInfo->lowerBoundDist, host_queryInfo->lowerBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);//for imp2
		free(d2h_queryInfo);

}

void TSGPUManager::update_windowQueryInfo_upperAndLowerBound_fromHostToDevice(WindowQueryInfo* device_queryInfo, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist){

		int size = new_upperBoundDist.size();
		WindowQueryInfo* d2h_queryInfo = (WindowQueryInfo*) malloc(sizeof(WindowQueryInfo));
		cudaMemcpy(d2h_queryInfo, device_queryInfo, sizeof(WindowQueryInfo), cudaMemcpyDeviceToHost);

		cudaMemcpy(d2h_queryInfo->upperBoundDist, new_upperBoundDist.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->lowerBoundDist, new_lowerBoundDist.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

		free(d2h_queryInfo);
}

void TSGPUManager::update_windowQueryInfo_entry_upperAndLowerBound(int queryId, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist){

	//update device
	update_windowQueryInfo_upperAndLowerBound_fromHostToDevice(this->d_windowQuery_info_set[queryId], new_upperBoundDist,  new_lowerBoundDist);
	//update host
	for(int i=0;i<this->h_windowQuery_info_set[queryId]->numOfDimensionToSearch;i++){
		h_windowQuery_info_set[queryId]->upperBoundDist[i] = new_upperBoundDist[i];
		h_windowQuery_info_set[queryId]->lowerBoundDist[i] = new_lowerBoundDist[i];
	}

}



/**
 * gqi_data_set: data area part to update groupquery info
 * gqi_data_sliding_indicator: it labels the starting position of "new" data compared with previous existing data (after and inclusive this postion, all data are NEW)
 */
void TSGPUManager::depressed_update_ContQueryInfo_set(vector<vector<float> > gqi_data_set,vector<int> gqi_data_newData_indicator){

	depressed_reset_winQuery_ACDUpdate_forLBKeogh();
	for(int i=0;i<gqi_data_set.size();i++){
		depressed_update_ContQueryInfo(i,gqi_data_set[i],gqi_data_newData_indicator[i]);
	}

	depressed_update_GroupQueryInfo_vec_fromHostToDevice( this->h_groupQuery_info_set, this->d_groupQuery_info_set );
	if(!(this->depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled)){
		depressed_update_winQuery_ACDUpdate_forLBKeogh_fromHostToDevice();
		//print_ACDUpdate_forLBKeogh();
	}

}

/**
 * reset the bound as zero after the function labelReset_windowQuery_lowerBound(windowQuery_id) with batch method
 *
 * Refer to:
 * void TSGPUManager::reset_windowQuery_lowerBound(int windowQuery_info_id){

	int wqi_start = windowQuery_info_id*this->getMaxWindowFeatureNumber();
	int wqi_end = wqi_start+this->getMaxWindowFeatureNumber();

	this->h_windowQuery_LBKeogh_updatedLabel[windowQuery_info_id]=true;
	thrust::fill(this->d_windowQuery_lowerBound.begin()+wqi_start,this->d_windowQuery_lowerBound.begin()+wqi_end, 0);//clear the Count&ACD table of this window query//
	if(enhancedLowerBound){
		thrust::fill(this->d_windowQuery_lowerBound_q2d.begin()+wqi_start,this->d_windowQuery_lowerBound_q2d.begin()+wqi_end, 0);//clear the Count&ACD table of this window query//
	}
}
 */
void TSGPUManager::dev_batch_reset_windowQuery_lowerBound(){

	int bklNum = d_windowQuery_LBKeogh_updatedLabel.size();

	 kernel_batch_labelReset_windowQuery_lowerBound<<<bklNum,THREAD_PER_BLK>>>(
			raw_pointer_cast(d_windowQuery_LBKeogh_updatedLabel.data()),
			this->enhancedLowerBound,//whether to update the lowerbound
			0,//set as 0
			raw_pointer_cast(this->d_windowQuery_lowerBound.data()),
			raw_pointer_cast(this->d_windowQuery_lowerBound_q2d.data())
		);
}

/**
//to re-calculate lower and uppper lower bound or effected window queries after new point add to the queries
//block number is equal to the maximum number of windows queries, i.e. length of windowQuery_LBKeogh_updatedLabel
* to replace the following functions:*
* ## to compute the lower and upper distance bound
*
* initQueryItem_LBKeogh(sc_band, h_gqi->data,
				i, groupQuery_maxdimension,
				this->winDim, h_qi->keyword,
				new_upwardDistanceBound, new_downwardDistanceBound);

void initQueryItem_LBKeogh(int sc_band,	float* queryData,
		int queryData_start, int queryData_len,
		int numOfDimensionToSearch,	float* keywords,
		vector<float>& upwardDistanceBound, vector<float>& downwardDistanceBound) {

	for (int i = 0; i < numOfDimensionToSearch; i++) {
		int j = (queryData_start + i - sc_band <= 0) ?
				0 : (queryData_start + i - sc_band);
		int s = (queryData_start + i + sc_band >= (queryData_len - 1)) ?
				(queryData_len - 1) : (queryData_start + i + sc_band);

		float up = -(float) INT_MAX; //find maximum value within [i-r,i+r]
		float down = (float) INT_MAX; //find minimum value within [i-r,i+r]

		for (; j <= s; j++) {
			if (up < queryData[j]) {
				up = queryData[j];
			}
			if (down > queryData[j]){
				down = queryData[j];
			}
		}

		up = up - keywords[i];
		down = keywords[i] - down;

		assert(up >= 0 && down >= 0);
		upwardDistanceBound[i] = up;
		downwardDistanceBound[i] = down;

	}
}
*
*
* ## to update to the GPU memeory
*
* void TSGPUManager::update_windowQueryInfo_entry_upperAndLowerBound(int queryId, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist){

	//update device
	update_windowQueryInfo_upperAndLowerBound_fromHostToDevice(this->d_windowQuery_info_set[queryId], new_upperBoundDist,  new_lowerBoundDist);
	//update host
	for(int i=0;i<this->h_windowQuery_info_set[queryId]->numOfDimensionToSearch;i++){
		h_windowQuery_info_set[queryId]->upperBoundDist[i] = new_upperBoundDist[i];
		h_windowQuery_info_set[queryId]->lowerBoundDist[i] = new_lowerBoundDist[i];
	}
}
 */

void TSGPUManager::dev_batch_reset_windowQueryInfo_entry_upperAndLowerBound(){

	int bklNum = d_windowQuery_LBKeogh_updatedLabel.size();

	kernel_batch_labelReset_windowQueryInfo_entry_upperAndLowerBound<<<bklNum,THREAD_PER_BLK>>>(
			raw_pointer_cast(d_windowQuery_LBKeogh_updatedLabel.data()),
			sc_band,
			raw_pointer_cast(d_windowQuery_info_set.data()),//output:
			raw_pointer_cast(d_groupQuery_info_set.data())//output
		);
}

/**
 * gqi_data_set: data area part to update groupquery info
 * gqi_data_sliding_indicator: it labels the starting position of "new" data compared with previous existing data (after and inclusive this postion, all data are NEW)
 */
void TSGPUManager::update_ContQueryInfo_set(vector<vector<float> > gqi_data_set,vector<int> gqi_data_newData_indicator){

	reset_windowQuery_LBKeogh_updateCtrl();

	//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo()");
	for(int i=0;i<gqi_data_set.size();i++){
		update_ContQueryInfo(i,gqi_data_set[i],gqi_data_newData_indicator[i]);
	}
	//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():forloop:update_ContQueryInfo()");

	//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():update_GroupQueryInfo_vec_dataAndStartWindId_fromHostToDevice()");
	update_GroupQueryInfo_vec_dataAndStartWindId_fromHostToDevice(this->h_groupQuery_info_set, this->d_groupQuery_info_set );
	//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():update_GroupQueryInfo_vec_dataAndStartWindId_fromHostToDevice()");

	this->d_windowQuery_LBKeogh_updatedLabel=this->h_windowQuery_LBKeogh_updatedLabel;//update to device

	//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():dev_batch_reset_windowQueryInfo_entry_upperAndLowerBound()");
	dev_batch_reset_windowQueryInfo_entry_upperAndLowerBound();//for imp
	//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():dev_batch_reset_windowQueryInfo_entry_upperAndLowerBound()");

	//profile_clockTime_start("conf_contQuery_nextStep():update_ContQueryInfo_set():dev_reset_windowQuery_lowerBound()");
	dev_batch_reset_windowQuery_lowerBound();//for reset  the bound as zero after the function labelReset_windowQuery_lowerBound(windowQuery_id) with batch method
	//profile_clockTime_end("conf_contQuery_nextStep():update_ContQueryInfo_set():dev_reset_windowQuery_lowerBound()");
	//update labeled window queries
}

void TSGPUManager::depressed_reset_TSGPUMananger_forGroupQuery(){
		thrust::fill(depressed_d_groupQuery_unfinished.begin(),depressed_d_groupQuery_unfinished.end(),true);
		thrust::fill(d_groupQuerySet_topkResults.begin(),d_groupQuerySet_topkResults.end(),CandidateEntry());
		thrust::fill(d_groupQuerySet_topkResults_size.begin(),d_groupQuerySet_topkResults_size.end(),0);
		thrust::fill(d_groupQuerySet_items_lowerBoundThreshold.begin(),d_groupQuerySet_items_lowerBoundThreshold.end(),0);
		thrust::fill(d_groupQuerySet_lowerBound.begin(),d_groupQuerySet_lowerBound.end(),INITIAL_LABEL_VALUE);
	}


/**
 * TODO:
 * re-initialize the memory for next step prediction. However, keep the d_groupQuerySet_topkResults record the old information
 * to estimate the threshold for new topk query
 */
void TSGPUManager::reset_TSGPUMananger_forGroupQuery(){
		//thrust::fill(d_groupQuerySet_topkResults.begin(),d_groupQuerySet_topkResults.end(),CandidateEntry());
		thrust::fill(d_groupQuerySet_items_lowerBoundThreshold.begin(),d_groupQuerySet_items_lowerBoundThreshold.end(),0);
		thrust::fill(d_groupQuerySet_lowerBound.begin(),d_groupQuerySet_lowerBound.end(),INITIAL_LABEL_VALUE);
	}


void TSGPUManager::depressed_init_winQuery_ACDUpdate_forLBKeogh(){
	//note: in the first step of continuous query, we do top-k query directly without
	//considering the update, so the initial value is false.
	this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = false;
	depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec.resize(depressed_gpuManager.getTotalnumOfQuery(),false);
	depressed_h_winQuery_LBKeogh_ACDUpdate_subCtrlVec.resize(depressed_gpuManager.getTotalnumOfQuery(),false);

	depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec.resize(depressed_gpuManager.getTotalnumOfQuery()*sc_band,0);
	depressed_h_winQuery_LBKeogh_ACDUpdate_valueVec.resize(depressed_gpuManager.getTotalnumOfQuery()*sc_band,0);

}

void TSGPUManager::depressed_reset_winQuery_ACDUpdate_forLBKeogh(){

	this->depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl = true;

	thrust::fill(depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec.begin(),depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec.end(),false);
	thrust::fill(depressed_h_winQuery_LBKeogh_ACDUpdate_subCtrlVec.begin(),depressed_h_winQuery_LBKeogh_ACDUpdate_subCtrlVec.end(),false);

	thrust::fill(depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec.begin(),depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec.end(),0);
	thrust::fill(depressed_h_winQuery_LBKeogh_ACDUpdate_valueVec.begin(),depressed_h_winQuery_LBKeogh_ACDUpdate_valueVec.end(),0);


}

void TSGPUManager::reset_windowQuery_LBKeogh_updateCtrl(){

	thrust::fill(this->d_windowQuery_LBKeogh_updatedLabel.begin(),this->d_windowQuery_LBKeogh_updatedLabel.end(),false);
	thrust::fill(this->h_windowQuery_LBKeogh_updatedLabel.begin(),this->h_windowQuery_LBKeogh_updatedLabel.end(),false);
}

void TSGPUManager::depressed_update_winQuery_ACDUpdate_forLBKeogh_fromHostToDevice(){
	depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec = depressed_h_winQuery_LBKeogh_ACDUpdate_subCtrlVec;
	depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec = depressed_h_winQuery_LBKeogh_ACDUpdate_valueVec;
}




//================================private function


void TSGPUManager::depressed_update_GroupQueryInfo_fromHostToDevice(GroupQuery_info* host_groupQueryInfo, GroupQuery_info* device_groupQueryInfo){


	cudaMemcpy(&(device_groupQueryInfo->groupId),&(host_groupQueryInfo->groupId),sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(&(device_groupQueryInfo->blade_id),&(host_groupQueryInfo->blade_id),sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(&(device_groupQueryInfo->startWindowId),&(host_groupQueryInfo->startWindowId),sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(&(device_groupQueryInfo->item_num),&(host_groupQueryInfo->item_num),sizeof(int),cudaMemcpyHostToDevice);

	int itemNum = host_groupQueryInfo->item_num;
	int queryLen = host_groupQueryInfo->getMaxQueryLen();


	GroupQuery_info* d2h_groupQueryInfo = (GroupQuery_info*) malloc(sizeof(GroupQuery_info));
	cudaMemcpy(d2h_groupQueryInfo, device_groupQueryInfo, sizeof(GroupQuery_info), cudaMemcpyDeviceToHost);

	cudaMemcpy((d2h_groupQueryInfo->item_dimensions_vec),(host_groupQueryInfo->item_dimensions_vec),sizeof(int)*itemNum,cudaMemcpyHostToDevice);
	cudaMemcpy((d2h_groupQueryInfo->depressed_item_query_finished),(host_groupQueryInfo->depressed_item_query_finished),sizeof(bool)*itemNum,cudaMemcpyHostToDevice);
	cudaMemcpy(d2h_groupQueryInfo->data,host_groupQueryInfo->data,sizeof(float)*queryLen,cudaMemcpyHostToDevice);

	free(d2h_groupQueryInfo);
}


void TSGPUManager::update_GroupQueryInfo_dataAndStartWindId_fromHostToDevice(GroupQuery_info* host_groupQueryInfo, GroupQuery_info* device_groupQueryInfo){


	//cudaMemcpy(&(device_groupQueryInfo->groupId),&(host_groupQueryInfo->groupId),sizeof(int),cudaMemcpyHostToDevice);//for imp3
	//cudaMemcpy(&(device_groupQueryInfo->blade_id),&(host_groupQueryInfo->blade_id),sizeof(int),cudaMemcpyHostToDevice);//for imp3
	cudaMemcpy(&(device_groupQueryInfo->startWindowId),&(host_groupQueryInfo->startWindowId),sizeof(int),cudaMemcpyHostToDevice);
	//cudaMemcpy(&(device_groupQueryInfo->item_num),&(host_groupQueryInfo->item_num),sizeof(int),cudaMemcpyHostToDevice);//for imp3

	//int itemNum = host_groupQueryInfo->item_num;
	int queryLen = host_groupQueryInfo->getMaxQueryLen();


	GroupQuery_info* d2h_groupQueryInfo = (GroupQuery_info*) malloc(sizeof(GroupQuery_info));
	cudaMemcpy(d2h_groupQueryInfo, device_groupQueryInfo, sizeof(GroupQuery_info), cudaMemcpyDeviceToHost);

	//cudaMemcpy((d2h_groupQueryInfo->item_dimensions_vec),(host_groupQueryInfo->item_dimensions_vec),sizeof(int)*itemNum,cudaMemcpyHostToDevice);
	//cudaMemcpy((d2h_groupQueryInfo->depressed_item_query_finished),(host_groupQueryInfo->depressed_item_query_finished),sizeof(bool)*itemNum,cudaMemcpyHostToDevice);
	cudaMemcpy(d2h_groupQueryInfo->data,host_groupQueryInfo->data,sizeof(float)*queryLen,cudaMemcpyHostToDevice);

	free(d2h_groupQueryInfo);
}




void TSGPUManager::copy_GroupQueryInfo_vec_FromHostToDevice(  host_vector<GroupQuery_info*>& hvec, device_vector<GroupQuery_info*>& _dvec )
{

	_dvec.reserve(hvec.size());
	for ( int i = 0; i < hvec.size(); i++ )
	{
		GroupQuery_info* host_gqi = hvec[i];
		int itemNum = host_gqi->item_num;
		int queryLen = host_gqi->getMaxQueryLen();

		// copy class to gpu
		GroupQuery_info* dev_gqi;
		cudaMalloc( (void **) &dev_gqi, sizeof(GroupQuery_info) );
		cudaMemcpy(dev_gqi, host_gqi, sizeof(GroupQuery_info), cudaMemcpyHostToDevice);




		int *item_dimensions_vec;
		bool* item_query_finished;
		float *data;

		cudaMalloc((void **) &item_dimensions_vec,sizeof(int)*itemNum);
		cudaMalloc((void **) &item_query_finished,sizeof(bool)*itemNum);
		cudaMalloc((void **) &data,sizeof(float)*queryLen);


		cudaMemcpy(item_dimensions_vec,host_gqi->item_dimensions_vec,sizeof(int)*itemNum,cudaMemcpyHostToDevice);
		cudaMemcpy(item_query_finished,host_gqi->depressed_item_query_finished,sizeof(bool)*itemNum,cudaMemcpyHostToDevice);
		cudaMemcpy(data,host_gqi->data,sizeof(float)*queryLen,cudaMemcpyHostToDevice);


		//copy pointer to GPU
		cudaMemcpy(&(dev_gqi->item_dimensions_vec),&item_dimensions_vec,sizeof(int *),cudaMemcpyHostToDevice);
		cudaMemcpy(&(dev_gqi->depressed_item_query_finished),&item_query_finished,sizeof(bool *),cudaMemcpyHostToDevice);
		cudaMemcpy(&(dev_gqi->data),&data,sizeof(float *),cudaMemcpyHostToDevice);
		_dvec.push_back(dev_gqi);

	}

}

void TSGPUManager::free_GroupQueryInfo_device(device_vector<GroupQuery_info*>& _dvec ){

	for ( int i = 0; i < _dvec.size(); i++ ){
			// copy class reference to cpu
			GroupQuery_info* dev_gqi = _dvec[i];

			int *item_dimensions_vec;
			cudaMemcpy(&item_dimensions_vec,&(dev_gqi->item_dimensions_vec),sizeof(int*),cudaMemcpyDeviceToHost);
			cudaFree(item_dimensions_vec);

			bool* item_query_finished;
			cudaMemcpy(&item_query_finished,&(dev_gqi->depressed_item_query_finished),sizeof(bool *), cudaMemcpyDeviceToHost);
			cudaFree(item_query_finished);

			float *data;
			cudaMemcpy(&data,&(dev_gqi->data),sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(data);

			cudaFree(dev_gqi);
	}
	_dvec.clear();
	_dvec.shrink_to_fit();
}


void TSGPUManager::copy_windowQueryInfo_vec_fromHostToDevice( host_vector<WindowQueryInfo*>& hvec, device_vector<WindowQueryInfo*>& _dvec)
{
	_dvec.reserve(hvec.size());
	for ( int i = 0; i < hvec.size(); i++ )
	{
		WindowQueryInfo* hostQueryInfo = hvec[i];
		int size = hostQueryInfo->numOfDimensionToSearch;


		// copy class to gpu
		WindowQueryInfo* devQueryInfo;
		cudaMalloc( (void **) &devQueryInfo, sizeof(WindowQueryInfo) );
		cudaMemcpy(devQueryInfo, hostQueryInfo, sizeof(WindowQueryInfo), cudaMemcpyHostToDevice);

		// copy real data to gpu
		//int *searchDim;
		//float *distanceFunc;
		//int *upperBoundSearch;
		//int *lowerBoundSearch;
		//int2 *lastPos;
		float *keyword;
		//float *dimWeight;
		float *upperBoundDist;
		float *lowerBoundDist;

		//cudaMalloc((void **) &searchDim, sizeof(int) * size);
		//cudaMalloc((void **) &distanceFunc, sizeof(int) * size);
		//cudaMalloc((void **) &upperBoundSearch, sizeof(int) * size);
		//cudaMalloc((void **) &lowerBoundSearch, sizeof(int) * size);
		//cudaMalloc((void **) &lastPos, sizeof(int2) * size);
		cudaMalloc((void **) &keyword, sizeof(float) * size);
		//cudaMalloc((void **) &dimWeight, sizeof(float) * size);
		cudaMalloc((void **) &upperBoundDist, sizeof(float) * size);
		cudaMalloc((void **) &lowerBoundDist, sizeof(float) * size);


		//cudaMemcpy(searchDim, hostQueryInfo->depressed_searchDim, sizeof(int) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(distanceFunc, hostQueryInfo->depressed_distanceFunc, sizeof(float) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(upperBoundSearch, hostQueryInfo->depressed_upperBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(lowerBoundSearch, hostQueryInfo->depressed_lowerBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(lastPos, hostQueryInfo->depressed_lastPos, sizeof(int2) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(keyword, hostQueryInfo->keyword, sizeof(float) * size, cudaMemcpyHostToDevice);
		//cudaMemcpy(dimWeight, hostQueryInfo->depressed_dimWeight, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(upperBoundDist, hostQueryInfo->upperBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lowerBoundDist, hostQueryInfo->lowerBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);


		// copy pointer to gpu
		//cudaMemcpy(&(devQueryInfo->depressed_searchDim), &searchDim, sizeof(int *), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(devQueryInfo->depressed_distanceFunc), &distanceFunc, sizeof(float *), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(devQueryInfo->depressed_upperBoundSearch), &upperBoundSearch, sizeof(int *), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(devQueryInfo->depressed_lowerBoundSearch), &lowerBoundSearch, sizeof(int *), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(devQueryInfo->depressed_lastPos), &lastPos, sizeof(int2 *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->keyword), &keyword, sizeof(float *), cudaMemcpyHostToDevice);
		//cudaMemcpy(&(devQueryInfo->depressed_dimWeight), &dimWeight, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->upperBoundDist), &upperBoundDist, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->lowerBoundDist), &lowerBoundDist, sizeof(float *), cudaMemcpyHostToDevice);

		_dvec.push_back(devQueryInfo);
	}

}


void TSGPUManager::depressed_copy_windowQueryInfo_vec_fromHostToDevice( host_vector<WindowQueryInfo*>& hvec, device_vector<WindowQueryInfo*>& _dvec)
{

	for ( int i = 0; i < hvec.size(); i++ )
	{
		WindowQueryInfo* hostQueryInfo = hvec[i];
		int size = hostQueryInfo->numOfDimensionToSearch;


		// copy class to gpu
		WindowQueryInfo* devQueryInfo;
		cudaMalloc( (void **) &devQueryInfo, sizeof(WindowQueryInfo) );
		cudaMemcpy(devQueryInfo, hostQueryInfo, sizeof(WindowQueryInfo), cudaMemcpyHostToDevice);

		// copy real data to gpu
		int *searchDim;
		float *distanceFunc;
		int *upperBoundSearch;
		int *lowerBoundSearch;
		int2 *lastPos;
		float *keyword;
		float *dimWeight;
		float *upperBoundDist;
		float *lowerBoundDist;

		cudaMalloc((void **) &searchDim, sizeof(int) * size);
		cudaMalloc((void **) &distanceFunc, sizeof(int) * size);
		cudaMalloc((void **) &upperBoundSearch, sizeof(int) * size);
		cudaMalloc((void **) &lowerBoundSearch, sizeof(int) * size);
		cudaMalloc((void **) &lastPos, sizeof(int2) * size);
		cudaMalloc((void **) &keyword, sizeof(float) * size);
		cudaMalloc((void **) &dimWeight, sizeof(float) * size);
		cudaMalloc((void **) &upperBoundDist, sizeof(float) * size);
		cudaMalloc((void **) &lowerBoundDist, sizeof(float) * size);


		cudaMemcpy(searchDim, hostQueryInfo->depressed_searchDim, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(distanceFunc, hostQueryInfo->depressed_distanceFunc, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(upperBoundSearch, hostQueryInfo->depressed_upperBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lowerBoundSearch, hostQueryInfo->depressed_lowerBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lastPos, hostQueryInfo->depressed_lastPos, sizeof(int2) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(keyword, hostQueryInfo->keyword, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(dimWeight, hostQueryInfo->depressed_dimWeight, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(upperBoundDist, hostQueryInfo->upperBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lowerBoundDist, hostQueryInfo->lowerBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);


		// copy pointer to gpu
		cudaMemcpy(&(devQueryInfo->depressed_searchDim), &searchDim, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_distanceFunc), &distanceFunc, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_upperBoundSearch), &upperBoundSearch, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_lowerBoundSearch), &lowerBoundSearch, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_lastPos), &lastPos, sizeof(int2 *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->keyword), &keyword, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_dimWeight), &dimWeight, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->upperBoundDist), &upperBoundDist, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->lowerBoundDist), &lowerBoundDist, sizeof(float *), cudaMemcpyHostToDevice);

		_dvec.push_back(devQueryInfo);
	}

}


void TSGPUManager::free_windowQueryInfo_vec_onDevice(  device_vector<WindowQueryInfo*>& _dvec ){

	for ( int i = 0; i < _dvec.size(); i++ )
		{
			// copy class reference to cpu
			WindowQueryInfo* devQueryInfo = _dvec[i];

			//int *searchDim;
			//cudaMemcpy(&searchDim,&(devQueryInfo->depressed_searchDim), sizeof(int *), cudaMemcpyDeviceToHost);
			//cudaFree(searchDim);

			//float *distanceFunc;
			//cudaMemcpy(&distanceFunc,&(devQueryInfo->depressed_distanceFunc), sizeof(float *), cudaMemcpyDeviceToHost);
			//cudaFree(distanceFunc);

			//int *upperBoundSearch;
			//cudaMemcpy(&upperBoundSearch,&(devQueryInfo->depressed_upperBoundSearch), sizeof(int *), cudaMemcpyDeviceToHost);
			//cudaFree(upperBoundSearch);

			//int *lowerBoundSearch;
			//cudaMemcpy(&lowerBoundSearch,&(devQueryInfo->depressed_lowerBoundSearch), sizeof(int *), cudaMemcpyDeviceToHost);
			//cudaFree(lowerBoundSearch);

			//int2 *lastPos;
			//cudaMemcpy(&lastPos,&(devQueryInfo->depressed_lastPos), sizeof(int2 *), cudaMemcpyDeviceToHost);
			//cudaFree(lastPos);

			float *keyword;
			cudaMemcpy(&keyword,&(devQueryInfo->keyword), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(keyword);

			//float *dimWeight;
			//cudaMemcpy(&dimWeight,&(devQueryInfo->depressed_dimWeight), sizeof(float *), cudaMemcpyDeviceToHost);
			//cudaFree(dimWeight);

			float *upperBoundDist;
			cudaMemcpy(&upperBoundDist,&(devQueryInfo->upperBoundDist), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(upperBoundDist);

			float *lowerBoundDist;
			cudaMemcpy(&lowerBoundDist,&(devQueryInfo->lowerBoundDist), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(lowerBoundDist);


			cudaFree( devQueryInfo );
		}

}


void TSGPUManager::depressed_free_windowQueryInfo_vec_onDevice(  device_vector<WindowQueryInfo*>& _dvec ){

	for ( int i = 0; i < _dvec.size(); i++ )
		{
			// copy class reference to cpu
			WindowQueryInfo* devQueryInfo = _dvec[i];

			int *searchDim;
			cudaMemcpy(&searchDim,&(devQueryInfo->depressed_searchDim), sizeof(int *), cudaMemcpyDeviceToHost);
			cudaFree(searchDim);

			float *distanceFunc;
			cudaMemcpy(&distanceFunc,&(devQueryInfo->depressed_distanceFunc), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(distanceFunc);

			int *upperBoundSearch;
			cudaMemcpy(&upperBoundSearch,&(devQueryInfo->depressed_upperBoundSearch), sizeof(int *), cudaMemcpyDeviceToHost);
			cudaFree(upperBoundSearch);

			int *lowerBoundSearch;
			cudaMemcpy(&lowerBoundSearch,&(devQueryInfo->depressed_lowerBoundSearch), sizeof(int *), cudaMemcpyDeviceToHost);
			cudaFree(lowerBoundSearch);

			int2 *lastPos;
			cudaMemcpy(&lastPos,&(devQueryInfo->depressed_lastPos), sizeof(int2 *), cudaMemcpyDeviceToHost);
			cudaFree(lastPos);

			float *keyword;
			cudaMemcpy(&keyword,&(devQueryInfo->keyword), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(keyword);

			float *dimWeight;
			cudaMemcpy(&dimWeight,&(devQueryInfo->depressed_dimWeight), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(dimWeight);

			float *upperBoundDist;
			cudaMemcpy(&upperBoundDist,&(devQueryInfo->upperBoundDist), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(upperBoundDist);

			float *lowerBoundDist;
			cudaMemcpy(&lowerBoundDist,&(devQueryInfo->lowerBoundDist), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(lowerBoundDist);


			cudaFree( devQueryInfo );
		}

}



void TSGPUManager::update_WindowQueryInfo(host_vector<WindowQueryInfo*> queryInfo_vec){

	for(int i=0;i<this->h_groupQuery_info_set.size();i++){

		int d_windowQuery_id = h_groupQuery_info_set[i]->startWindowId + i*this->windowNum_perGroup;
		depressed_gpuManager.update_windowQuery_entryAndTable(queryInfo_vec[i],d_windowQuery_id);

	}
}



void TSGPUManager::print_d_groupQuerySet_candidates_scan_byEndIdx(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
		device_vector<int>& d_groupQuerySet_candidates_scan_endIdx) {

	host_vector<CandidateEntry> h_cs = d_groupQuerySet_candidates_scan;
	host_vector<int> h_cs_endidx = d_groupQuerySet_candidates_scan_endIdx;

	cout << " with debug purpose print *_scan++++++++++++++++\n";
	for (int i = 0; i < h_cs_endidx.size(); i++) {

		int s = (i == 0) ? 0 : h_cs_endidx[i - 1];
		int e = h_cs_endidx[i];
		//if(i==5){//with debug purpose//
		cout << "query item { " << i << " } with candiate number["<<e-s<<"]"<<endl;

		for (int j = s; j < e; j++) {
			//if(h_cs[j].feature_id>=63220||h_cs[j].feature_id==9261){//

			cout << "query item { " << i << " } with specif query item and feature id" << endl;
			//cout<<"  j:"<<j<<" s:"<<s<<" e:"<<e<<endl;
			h_cs[j].print();

			//}//end
		  }
		//}//end

	}
	cout << " end with debug purpose print *_scan\n";
}

void TSGPUManager::print_d_groupQuerySet_candidates_scan_bySize(
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
		device_vector<int>& d_groupQuerySet_candidates_scan_size) {

	host_vector<CandidateEntry> h_cs = d_groupQuerySet_candidates_scan;
	host_vector<int> h_cs_size = d_groupQuerySet_candidates_scan_size;

	cout << " with debug purpose print *_scan++++++++++++++++\n";
	int s=0, e=0;
	for (int i = 0; i < h_cs_size.size(); i++) {
		//int s = (i == 0) ? 0 : h_cs_size[i - 1];
		e = s+ h_cs_size[i];

		for (int j = s; j < e; j++) {
			//if(((h_cs[j].feature_id>=41723&&h_cs[j].feature_id<=41725)&&(i>=2718&&i<=2718))||(h_cs[j].feature_id==43840&&i==2856)){//feature id 43840
			cout << "query item { " << i << " } with specif query item and feature id" << endl;
			//cout<<" j:"<<j<<" s:"<<s<<" e:"<<e<<endl;
			h_cs[j].print();
			//}
		}
		s = e;

	}
	cout << " end with debug purpose print *_scan\n";
}

void TSGPUManager::print_d_groupQuerySet_candidates_scan_size( device_vector<int>& d_groupQuerySet_candidates_scan_size){

	//for exp
		//cout<<"with debug purpose print dev_output_groupQuery_candidates_size()"<<endl;
		host_vector<int> h_groupQuerySet_candidates_scan_size= d_groupQuerySet_candidates_scan_size;
		double sum=0;
		double sum_sqaure=0;
		for(int i=0;i<h_groupQuerySet_candidates_scan_size.size();i++){
			int size = h_groupQuerySet_candidates_scan_size[i];
			sum+=size;
			sum_sqaure+= size*size;
			cout<<" group query item ["<<i<<"] candidate size: {"<<h_groupQuerySet_candidates_scan_size[i]<<"}"<<endl;
		}
		double mean = (sum*1.0)/h_groupQuerySet_candidates_scan_size.size();
		double mean_square = (sum_sqaure*1.0)/h_groupQuerySet_candidates_scan_size.size();
		double std_var = std::sqrt(mean_square - mean*mean);
		//cout<<"statistics of un-filtered items in d_groupQuerySet_candidates_scan_size:"<<endl
		//		<<" mean: "<<mean
		//		<<" mean_square:"<<mean_square
		//		<<" standard variance: "<<std_var
		//		<<endl;
		//end for exp
}



void TSGPUManager::print_d_groupQuerySet_candidates_verified(int topk,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates_verified,
		device_vector<int>& d_groupQuerySet_candidates_verified_size) {

	cout << " with debug purpose start print *_verified++++++++++++++++++\n";

	host_vector<CandidateEntry> h_ct = d_groupQuerySet_candidates_verified;
	host_vector<int> h_ct_size = d_groupQuerySet_candidates_verified_size;
	cout << " with debug purpose h_ct_size.size:" << h_ct_size.size() << endl;

	for (int i = 0; i < h_ct_size.size(); i++) {

		//if(i==5){//
		cout << " query item {" << i << " } h_ct_size:" << h_ct_size[i]
				<< endl;
		int s = i * topk;
		int e = s + h_ct_size[i];

		for (int j = s; j < e; j++) {

			cout << " query item {" << i << " } " << endl;
			h_ct[j].print();

		}
		//}//
	}
	cout << "end with debug purpose print *_top++++++++++++++++++++++++\n";

}

void TSGPUManager::print_d_groupQuery_info_set(){
	cout<< " start print print_d_groupQuery_info_set======================\n";
	cout<< " the GroupQuery_info is:"<<endl;
		for(int i=0;i<this->h_groupQuery_info_set.size();i++){
			this->h_groupQuery_info_set[i]->print();
		}
	cout<< " end print d_groupQuery_info_set ======================\n";
}

void TSGPUManager::print_d_windowQuery_info_set(){

	cout<< " start print d_windowQuery_info_set======================\n";
	cout<< " the windowQuery_info_set is:"<<endl;
	for(int i=0;i<this->h_windowQuery_info_set.size();i++){
			this->h_windowQuery_info_set[i]->print();
		}

	cout<< " end pprint d_windowQuery_info_set======================\n";
}

void TSGPUManager::print_d_groupQuerySet_topkResults(int topk) {
	cout<< " start print d_groupQuerySet_topkResults candidates======================\n";

	cout<< " the GroupQuery_info is:"<<endl;
	for(int i=0;i<this->h_groupQuery_info_set.size();i++){
		//if(i==5){//with debug purpose//
		this->h_groupQuery_info_set[i]->print();
		//}//
	}

	cout<<"with debug purpose d_groupQuerySet_topkResults.size():"<<d_groupQuerySet_topkResults.size()<<endl;
	host_vector<CandidateEntry> h_vc = this->d_groupQuerySet_topkResults;


	//cout<<"with debug purpose: print array of d_groupQuerySet_topkResults"<<endl;
	//for(int i=0;i<64;i++){
	//	h_vc[i].print();
	//}



	for (int i = 0; i < h_vc.size(); i++) {

		//if((i/topk==5)){//with debug purpose//
		if(i%topk==0){
			cout<<endl<<"===for query item:"<<i / topk<<"============"<<endl;
		}
		//if(i%topk==(topk-1)||i%topk==(topk-2)){//
		cout << " query item {" << i / topk << "}" << endl;

		h_vc[i].print();
		//}//end
		//}//

	}
	cout<< "end print d_groupQuerySet_topkResults candidates======================\n";
}

void TSGPUManager::depressed_print_windowQuery_LowerBound() {

	//
	cout << "with debug purpose print d_winQuery_lowerBound" << endl;
	host_vector<float> hwb = this->d_windowQuery_lowerBound;
	host_vector<QueryFeatureEnt> h_qf =
			depressed_gpuManager.get_d_query_feature_reference();
	printf("d_winQuery_lowerBound.size=%d\n",d_windowQuery_lowerBound.size());
	for (int i = 0; i < this->d_windowQuery_lowerBound.size(); i++) {

		printf("query item { %d } windows start id [ %d ] lowerbound  ( %f )\n",
				(i / depressed_gpuManager.getMaxFeatureNumber()) ,
				winDim * (i % depressed_gpuManager.getMaxFeatureNumber()) , hwb[i]);
		h_qf[i].print();
	}
	cout << "end with debug purpose print d_winQuery_lowerBound" << endl;
}

void TSGPUManager::print_windowQuery_lowerBound(){
	cout << "with debug purpose print d_winQuery_lowerBound" << endl;
	host_vector<float> hwb = this->d_windowQuery_lowerBound;
	printf("d_winQuery_lowerBound.size=%d MaxWindowFeatureNumber per windowQuery=%d\n",d_windowQuery_lowerBound.size(),this->getMaxWindowFeatureNumber());

	for (int i = 0; i < this->d_windowQuery_lowerBound.size(); i++) {

		  //if( ((i % this->getMaxWindowFeatureNumber())<32)&&((i / this->getMaxWindowFeatureNumber())==0)){
			printf("window query item { %d } windows id [ %d ] lowerbound  ( %.3f )\n",
					(i / this->getMaxWindowFeatureNumber()) ,
					 (i % this->getMaxWindowFeatureNumber()) , hwb[i]);
		//  }

	}
	cout << "end with debug purpose print d_winQuery_lowerBound" << endl;
}




void TSGPUManager::depressed_print_groupQuery_LowerBound() {

	cout << "with debug purpose print d_groupQuerySet_lowerBound" << endl;
	host_vector<groupQuery_boundEntry> h_gl = d_groupQuerySet_lowerBound;
	for (int i = 0; i < h_gl.size(); i++) {
		printf("query item %d  ts feature id %d  lowerbound %f\n",
				i / (winDim * depressed_gpuManager.getMaxFeatureNumber()),
				i % (winDim * depressed_gpuManager.getMaxFeatureNumber()), h_gl[i]);
	}
	cout << "end with debug purpose print d_groupQuerySet_lowerBound" << endl;

}

void TSGPUManager::print_groupQuery_LowerBound(){
	cout << "with debug purpose print d_groupQuerySet_lowerBound with d_groupQuerySet_lowerBound.size():"<<d_groupQuerySet_lowerBound.size() << endl;
		host_vector<groupQuery_boundEntry> h_gl = d_groupQuerySet_lowerBound;
		for (int i = 0; i < h_gl.size(); i++) {
			//if(i / (this->getMaxFeatureNumber())==2&&((i % this->getMaxFeatureNumber()<32)||(i % this->getMaxFeatureNumber()>63300))){//with debug purpose
			if(i / (this->getMaxFeatureNumber())==2&&((i % this->getMaxFeatureNumber()<32))){//with debug purpose
			printf("groupQuery item %d  ts feature id  %d  lowerbound %f\n",
					i / (this->getMaxFeatureNumber()),
					i % (this->getMaxFeatureNumber()), h_gl[i]);
			}
	}
	cout << "end with debug purpose print d_groupQuerySet_lowerBound" << endl;
}

void TSGPUManager::print_ACDUpdate_forLBKeogh() {

	cout << "with debug purpose print ACDUpdate_forLBKeogh" << endl;
	host_vector<bool> ak_ctrl = this->depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec;
	cout << "with debug purpose print d_winQuery_LBKeogh_ACDUpdate_subCtrlVec" << endl;
	for (int i = 0; i < ak_ctrl.size(); i++) {
		cout << "  winQuery ["<<i<<"]: " << ak_ctrl[i];
	}
	cout << endl;

	host_vector<float> ak_float = this->depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec;
	cout << "with debug purpose print d_winQuery_LBKeogh_ACDUpdate_valueVec" << endl;
	for (int i = 0; i < ak_float.size(); i++) {
		cout << " window query entry " << i / sc_band << " expansion value:"
				<< ak_float[i] << endl;
	}
}

void TSGPUManager::print_d_groupQuerySet_items_lowerBoundThreshold(){
	cout << "with debug purpose print d_groupQuerySet_items_lowerBoundThreshold" << endl;
	host_vector<float> h_qil = this->d_groupQuerySet_items_lowerBoundThreshold;

	for(int i=0;i<h_qil.size();i++)
	{
			cout<<" query item "<<i<<" with threashold:"<<h_qil[i]<<endl;

	}
}


void TSGPUManager::print_d_groupQuerySet_candidates_selected_idSet(int selectedCandidates_num, device_vector<int>& d_groupQuerySet_candidates_selected_idSet,
		device_vector<CandidateEntry>& d_groupQuerySet_candidates,device_vector<int>& d_groupQuerySet_candidates_endIdx){

	cout << "with debug purpose print_d_groupQuerySet_candidates_selected_idSet() with idSet size:"<<d_groupQuerySet_candidates_selected_idSet.size()
			<<"and query number:"<<d_groupQuerySet_candidates_endIdx.size()<< endl;
	host_vector<int> h_idSet = d_groupQuerySet_candidates_selected_idSet;
	host_vector<CandidateEntry> h_ce = d_groupQuerySet_candidates;
	host_vector<CandidateEntry> h_ce_selected(selectedCandidates_num * this->getTotalQueryItemNum());

	cout<<"print_d_groupQuerySet_candidates_selected_idSet(): print start"<<endl;
	for (int i = 0; i < d_groupQuerySet_candidates_endIdx.size(); i++) {
		//if(i==5){//with debug purpose//
		for (int j = 0; j < selectedCandidates_num; j++) {
			//if (h_idSet[i * selectedCandidates_num + j] < 0) {//with debug purpose
				cout << " query item{" << i << "}" << endl;
				cout << " h_idSet[" << i * selectedCandidates_num + j << "]:"
						<< h_idSet[i * selectedCandidates_num + j] << endl;
			//}
			h_ce_selected[i * selectedCandidates_num + j] = h_ce[h_idSet[i
					* selectedCandidates_num + j]];
			   cout<<"select item from d_groupQuerySet_candidates is:"<<endl;
			   h_ce_selected[i * selectedCandidates_num + j].print();
		}
		//}//end with debug purpose
	}
	cout<<"print_d_groupQuerySet_candidates_selected_idSet(): print end"<<endl;

}

void TSGPUManager::print_d_groupQuerySet_lowerBound_selected_idSet(int selNum,device_vector<int>& d_groupQuerySet_lowerBound_sel_idSet,
		device_vector<groupQuery_boundEntry>& d_groupQuerySet_lowerBound,device_vector<int>& d_groupQuerySet_lowerBound_endIdx){
	cout << "with debug purpose print_d_groupQuerySet_lowerBound_selected_idSet()"
				<< endl;

	host_vector<int> h_idSet = d_groupQuerySet_lowerBound_sel_idSet;
	host_vector<groupQuery_boundEntry> h_ce = d_groupQuerySet_lowerBound;
	host_vector<groupQuery_boundEntry> h_ce_selected(selNum * this->getTotalQueryItemNum());

	for (int i = 0; i < d_groupQuerySet_lowerBound_endIdx.size(); i++) {
			for (int j = 0; j < selNum; j++) {
				if (h_idSet[i * selNum + j] < 0) {
					cout << " query item{" << i << "}" << endl;
					cout << " h_idSet[" << i * selNum + j << "]:"
							<< h_idSet[i * selNum + j] << endl;
				}
				h_ce_selected[i * selNum + j] = h_ce[h_idSet[i * selNum + j]];
			}
		}
}




void TSGPUManager::profile_clockTime_start(string tag){
		struct timespec total_tim;
//		gettimeofday(&total_tim, NULL);
		clock_gettime( CLOCK_REALTIME, &total_tim);
		exec_timespec_set[prefix_timespec+tag]=total_tim;
}

/**
 * string tag:
 */
void TSGPUManager::profile_clockTime_end(string tag){
		map<string,struct timespec>::iterator time_itr;
		time_itr = exec_timespec_set.find(prefix_timespec+tag);
		if(time_itr==exec_timespec_set.end()){
			cout<<"error: time profiler tag:"<<prefix_timespec+tag<<" is undefined!!!"<<endl;
			exit(1);
		}

		struct timespec total_tim = exec_timespec_set[prefix_timespec+tag];
		double t0=total_tim.tv_sec+(total_tim.tv_nsec/BILLION);
		clock_gettime( CLOCK_REALTIME, &total_tim);
		double t1 = total_tim.tv_sec+(total_tim.tv_nsec/BILLION);

		if(exec_timeInfo_set.find(prefix_timespec+tag)==exec_timeInfo_set.end()){
			this->exec_timeInfo_set[prefix_timespec+tag] = t1-t0;
			this->exec_timeInfo_count_set[prefix_timespec+tag]=1;
		}else{
			this->exec_timeInfo_set[prefix_timespec+tag] += t1-t0;
			this->exec_timeInfo_count_set[prefix_timespec+tag]+=1;
		}

		exec_timespec_set.erase(time_itr);
}



void TSGPUManager::profile_cudaTime_start(string tag){
			if(this->exec_cudaEvent_t_set.size()==0){
				cudaProfilerStart();
			}
			cudaEvent_t cuda_start;
			cudaEventCreate(&cuda_start);

			cudaEventRecord(cuda_start, 0);
			this->exec_cudaEvent_t_set[prefix_cudatime+tag]=cuda_start;


	}


/**
 * string tag:
 */
void TSGPUManager::profile_cudaTime_end(string tag){
	      map<string, cudaEvent_t>::iterator cudaEvent_itr;
	      cudaEvent_itr = this->exec_cudaEvent_t_set.find(prefix_cudatime+tag);
	      if(cudaEvent_itr==this->exec_cudaEvent_t_set.end()){
	    	  cout<<"error: time profiler tag:"<<prefix_timespec+tag<<" is undefined!!!"<<endl;
	    	  exit(1);
	      }

	      cudaEvent_t cuda_stop;
	      cudaEventCreate(&cuda_stop);

	      cudaEventRecord(cuda_stop, 0);
	      cudaEventSynchronize(cuda_stop);

	      float cuda_elapsedTime=0;
	      cudaEvent_t cuda_start = exec_cudaEvent_t_set[prefix_cudatime+tag];
	      cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);


	      if(exec_timeInfo_set.find(prefix_cudatime+tag)==exec_timeInfo_set.end()){
	    	  this->exec_timeInfo_set[prefix_cudatime+tag] = cuda_elapsedTime/1000.0;
	    	  this->exec_timeInfo_count_set[prefix_cudatime+tag] = 1;
	      	}else{
	      		this->exec_timeInfo_set[prefix_cudatime+tag] += cuda_elapsedTime/1000.0;
		    	this->exec_timeInfo_count_set[prefix_cudatime+tag] += 1;
	      	}

	      exec_cudaEvent_t_set.erase(cudaEvent_itr);
	      if(exec_cudaEvent_t_set.size()==0){
	    	  cudaProfilerStop();
	      }



}

void TSGPUManager::print_proflingTime( ){//be here !!!
	cout<<"profiling time:"<<endl;

	for(std::map<string,double>::iterator it = exec_timeInfo_set.begin();it!=exec_timeInfo_set.end();it++){
		cout<<it->first<<" time: "<<it->second<<"  step count:"<<this->exec_timeInfo_count_set[it->first]<<endl;
	}
}

void TSGPUManager::print_proflingTime_perContStep(){

	if(this->enable_sum_unfiltered_candidates){
		double sum = this->sum_unfiltered_candidates;
		string holder = "A sum:exact_topkQuery_DTW_contNextStep()";
		string key = prefix_cudatime+holder;
		int count= this->exec_timeInfo_count_set[key];
		cout<<"continuous step count:"<<count<<endl;
		int gqNum = this->getGroupQueryNum();
		int total_gq_item_num=this->getTotalQueryItemNum();
		cout<<"profiling un-fitered candidates total:"<< sum<<endl;
		cout<<"profiling un-fitered candidates per prediction step:"<<sum/count<<endl;
		cout<<"profiling un-fitered candidates per prediction step per group query:"<<sum/count/gqNum<<endl;
		cout<<"profiling un-fitered candidates per prediction step per group query item:"<<sum/count/total_gq_item_num<<endl;
	}

	cout<<"profiling time per continuous step:"<<endl;

	for(std::map<string,double>::iterator it = exec_timeInfo_set.begin();it!=exec_timeInfo_set.end();it++){
			cout<<it->first<<" time: "<<it->second/this->exec_timeInfo_count_set[it->first]<<endl;
	}


}



