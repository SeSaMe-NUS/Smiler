/*
 * TSGPUManager.h
 *
 *  Created on: May 13, 2014
 *      Author: zhoujingbo
 */

#ifndef TSGPUMANAGER_H_
#define TSGPUMANAGER_H_

//for cuda files

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
using namespace thrust;


//for stl
#include <map>
using namespace std;

//for custom files
#include "TSGPUFunctions.h"
#include "../GPUKNN/GPUManager.h"
#include "UtlTSProcess.h"

/**
 * important assumption:
 * 1. all data in different groups have the same number of maxWinFeatureID, or their maxWinFeaturesIDs are similar and we select the maximum one.
 * 2. all the group of query have the same number of sliding window query, all sliding windows have the same length,
 * 	   -> this condition means all the group query should have the same number of maximum query length
 * 3. the minimum query dimensions of group query should be 2 time of the width of sliding windows
 *    create the sliding windows in reverse order (from tailed to head) for group query.
 * 4. all the group queries have the same number of query items
 */


/**
 *
 * TODO:
 * 1. modify bucket_topk, output the size of selected item
 * 2. give min and max for different query data area
 * 3. imporve the function dev_fastSelect_and_verify_topkCandidates(), we may select more candidates number which is larger than k and then use fast selection to get the topk candidates
 * 4. in exact_TopK_query_DTW_fastSelect(),
 *     dev_maintain_topkCandidates_mergeSort(
 *					topk,
 *					d_groupQuerySet_candidates_verified,
 *					d_groupQuerySet_candidates_scan_size);//will be changed back to d_groupQuerySet_candidates_verified_size
 */



class TSGPUManager {
public:
	//TSGPUManager(GPUSpecification& gpuSpec);
	TSGPUManager();

	virtual ~TSGPUManager();

//public function
public:
	void conf_dev_blade(const int bladeNum, const vector<float>& blade_data_vec, const vector<int>& blade_len_vec );
	void conf_dev_blade(vector<vector<float> >& bladeData_vec);
	void conf_dev_query(vector<GroupQuery_info*>& groupQuery_info_set,
			 const int windowNumber_perGroup,const int itemNum_perGroup,const int winDim, const int SC_Band);
	void depressed_conf_dev_index(GPUSpecification& windQuery_GpuSpec,const int winDim);
	//initialize and set query
	void depressed_conf_dev_query(vector<GroupQuery_info*>& gpuGroupQuery_info_set,
			vector<int>& upSearchExtreme_vec, vector<int>& downSearchExtreme_vec,
			 const int windowNumber_perGroup,const int itemNum_perGroup,const int SC_Band);

	void init_parameters_default();

	//void exact_TopK_query_DTW(int topk);
	void depressed_exact_topk_query_DTW_randomSelect_bucketUnit(int topk);
	void depressed_exact_TopK_query_DTW_randomSelect_bucketWidth(int topk);
	void depressed_exact_TopK_query_DTW_randomSelect(int topk, bool isBucketUnit);

	void depressed_exact_TopK_query_DTW_fastSelect_bucketUnit(int topk);
	void depressed_exact_TopK_query_DTW_fastSelect_bucketWidth(int topk);
	void depressed_exact_TopK_query_DTW_fastSelect(int topk,bool isBucketUnit);

	void exact_topkQuery_DTW_computeLowerBound();
	void exact_topkQuery_DTW_afterBoundAndThreshold(int topk);
	void exact_topkQuery_DTW(int topk);



	void exact_topkQuery_DTW_contFirstStep(int topk);
	void estimate_topkThreshold_contNextStep(int topk);
	void exact_topkQuery_DTW_contNextStep(int topk);
	void exact_topkQuery_DTW_contNextStep_withInitThreshold(int topk);
	//=============for sliding index


	//auxiliary functions
	void getTopkResults(host_vector<CandidateEntry>& _topk_res){_topk_res = this->d_groupQuerySet_topkResults;}
	void print_d_windowQuery_info_set();
	void print_d_groupQuery_info_set();
	void print_d_groupQuerySet_topkResults(int topk);
	void print_d_groupQuerySet_items_lowerBoundThreshold();
	void print_d_groupQuerySet_candidates_selected_idSet(int selectedCandidates_num, device_vector<int>& d_groupQuerySet_candidates_selected_idSet,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,device_vector<int>& d_groupQuerySet_candidates_endIdx);
	void print_d_groupQuerySet_lowerBound_selected_idSet(int selNum,device_vector<int>& d_groupQuerySet_lowerBound_sel_idSet,
			device_vector<groupQuery_boundEntry>& d_groupQuerySet_lowerBound,device_vector<int>& d_groupQuerySet_lowerBound_endIdx);

	void profile_clockTime_start(string tag);
	void profile_clockTime_end(string tag);

	void profile_cudaTime_start(string tag);
	void profile_cudaTime_end(string tag);

	void print_proflingTime();
	void print_proflingTime_perContStep();


private:
	void depressed_dev_BidirectionExpansion_bucketUnit();
	void depressed_dev_compute_windowQuery_LowerBound_bucketUnit();

	void depressed_dev_BidirectionExpansion_bucket_exclusive();
	void depressed_dev_compute_windowQuery_LowerBound_bucket_inclusive();

	void dev_compute_windowQuery_lowerBound();
	void dev_compute_windowQuery_enhancedLowerBound();
	void dev_compute_TSData_upperLowerBound();

	void depressed_dev_compute_groupQuery_LowerBound();
	void dev_compute_groupQuery_lowerBound();
	void dev_compute_groupQuery_enhancedLowerBound();
	void depressed_dev_output_groupQuery_candidates(
			device_vector<CandidateEntry>& _d_groupQuerySet_candidates,
			device_vector<int>& _d_groupQuerySet_candidates_size);
	void dev_output_groupQuery_candidates(
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//output
			device_vector<int>& d_groupQuerySet_candidates_size//output
		);
	void dev_output_groupQuery_candidates(
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//output
			device_vector<int>& d_groupQuerySet_candidates_size,//output
			device_vector<int>& d_thread_threshold_endIdx
		);
	void depressed_dev_scan_verifyCandidates_fixedNum(int verified_candidates_num,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,
			device_vector<int>& d_groupQuerySet_candidates_size);
	void dev_scan_verifyCandidates_fixedNum(
			int verified_candidates_num,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
			device_vector<int>& d_groupQuerySet_candidates_size);
	void depressed_dev_scan_verifyCandidate(
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
			device_vector<int>& d_groupQuerySet_candidates_endIdx);
	void dev_scan_verifyCandidate(
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
			device_vector<int>& d_groupQuerySet_candidates_endIdx);
	void dev_scan_verifyCandidate_perThreadGenerated(
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input and output: caculate the distance
			device_vector<int>& d_groupQuerySet_candidate_threadThreshold_endIdx
			);


	void dev_select_candidates_mergeSort_fixedNum(
				int selected_candidates_num,
				int fixed_candidates_num,
				device_vector<CandidateEntry>& d_groupQuerySet_candidates,
				device_vector<int>& d_groupQuerySet_candidates_size,
				device_vector<CandidateEntry>& groupQuerySet_candidates_kSelected,
				device_vector<int>& groupQuerySet_candidates_kSelected_size);

	void dev_select_candidates_mergeSort_inPlace(
			int selected_candidates_num, //number of candidates want to select
			int fixed_candidates_num,//number of items for selection, the actual number may be samller than fixed_candidates_num, which is recorded in "d_groupQuerySet_candidates_size"
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,//input
			device_vector<int>& d_groupQuerySet_candidates_size,//input
			device_vector<CandidateEntry>& groupQuerySet_candidates_selected,//output
			device_vector<int>& groupQuerySet_candidates_selected_size//output:note this is size, not endIdx
			);

	bool dev_check_groupQuery_finished();
	void set_threshold_from_topkResults_byPermutation(int topk);
	void set_threshold_from_topkResults_byCopy(int topk);
	void dev_maintain_topkCandidates_mergeSort(int topk,
				device_vector<CandidateEntry>& d_groupQuerySet_candidates_kSelected,
				device_vector<int>& d_groupQuerySet_candidates_kSelected_size);


	//for topk with random selection

	//void dev_randomSelect_verify_candidates(int topk, device_vector<CandidateEntry>& groupQuerySet_candidates_kSelected);
	void depressed_dev_randomSelect_verify_candidates(int topk, int candidates_num,
			device_vector<CandidateEntry>& groupQuerySet_candidates_kSelected);
	void dev_select_candidates_random(int selectedCandidates_num,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,
			device_vector<int>& d_groupQuerySet_candidates_endIdx,
			device_vector<CandidateEntry>& groupQuerySet_candidates_kSelected,
			device_vector<int>& groupQuerySet_candidates_kSelected_size);

	void depressed_dev_randomSelect_and_Verify_topkCandidates(int topk,
			int verify_candidates_num,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
			device_vector<int>& d_groupQuerySet_candidates_scan_endIdx,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates_topk,
			device_vector<int>& d_groupQuerySet_candidates_topk_size);
	void depressed_init_topkThreshold_randomSelect(int topk);

	//for topk with fast selection
	void getMinMax_d_groupQuerySet_lowerBound(float& min, float& max);
	void init_topkThreshold_fastSelect(int topk);

	void getMinMax_groupQuerySet_candidates(device_vector<CandidateEntry>& d_groupQuerySet_candidates, CandidateEntry& min, CandidateEntry& max);
	void depressed_dev_select_candidates_fast(int selectedCandidates_num,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,
			device_vector<int>& d_groupQuerySet_candidates_startIdx,
			device_vector<int>& d_groupQuerySet_candidates_endIdx,
			device_vector<CandidateEntry>& groupQuerySet_candidates_selected //output
			);

	void dev_select_candidates_fast(int selectedCandidates_num,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates,
			device_vector<int>& d_groupQuerySet_candidates_startIdx,
			device_vector<int>& d_groupQuerySet_candidates_endIdx,
			device_vector<CandidateEntry>& groupQuerySet_candidates_selected //output
			);

	void dev_capIntVector(int cap,device_vector<int>& org, device_vector<int>& trans);
	void depressed_dev_fastSelect_and_verify_topkCandidates(int topk,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
			device_vector<int>& d_groupQuerySet_candidates_scan_size,
			device_vector<CandidateEntry>& d_groupQuerySet_candidates_topk,
			device_vector<int>& d_groupQuerySet_candidates_topk_size);


	void depressed_update_GroupQueryInfo_fromHostToDevice(GroupQuery_info* host_groupQueryInfo, GroupQuery_info* device_groupQueryInfo);
	void update_GroupQueryInfo_dataAndStartWindId_fromHostToDevice(GroupQuery_info* host_groupQueryInfo, GroupQuery_info* device_groupQueryInfo);
	void depressed_update_GroupQueryInfo_vec_fromHostToDevice( host_vector<GroupQuery_info*> hvec, device_vector<GroupQuery_info*> dvec );
	void update_GroupQueryInfo_vec_dataAndStartWindId_fromHostToDevice( host_vector<GroupQuery_info*> hvec, device_vector<GroupQuery_info*> dvec );
	void depressed_update_winQuery_LBKeogh_ACDUpdate(int new_window_start, int groupQueryInfo_id );


public:
	int getMaxDimensionsOfGroupQueries(vector<GroupQuery_info*>& gpuGroupQuery_info_set);
	int depressed_getMaxWindowFeatureNumber();
	int comp_maxWindowFeatureNumber();
	int getMaxWindowFeatureNumber(){return this->maxWindowFeatureNumber;}
	int depressed_getMaxFeatureNumber(){return (this->depressed_getMaxWindowFeatureNumber()*winDim);}
	int getMaxFeatureNumber(){return this->maxWindowFeatureNumber*winDim;}
	int getWinDim(){return winDim;}
	int getGroupQueryNum(){ return d_groupQuery_info_set.size();}
	int getGroupQueryMaxDimension() {return groupQuery_maxdimension;}
	int getQueryItemsNumberPerGroup(){return groupQuery_item_num;}
	int getTotalQueryItemNum(){return (this->getGroupQueryNum()) * (this->getQueryItemsNumberPerGroup());}
	int getBladeNum(){return ts_blade_num;}
//private function
private:
	void initMemeberVariables();
	void copy_GroupQueryInfo_vec_FromHostToDevice(  host_vector<GroupQuery_info*>& hvec, device_vector<GroupQuery_info*>& _dvec );
	void free_GroupQueryInfo_device(device_vector<GroupQuery_info*>& _dvec );

	void depressed_copy_windowQueryInfo_vec_fromHostToDevice( host_vector<WindowQueryInfo*>& hvec, device_vector<WindowQueryInfo*>& _dvec);
	void copy_windowQueryInfo_vec_fromHostToDevice( host_vector<WindowQueryInfo*>& hvec, device_vector<WindowQueryInfo*>& _dvec);
	void depressed_free_windowQueryInfo_vec_onDevice( device_vector<WindowQueryInfo*>& _dvec );
	void free_windowQueryInfo_vec_onDevice(  device_vector<WindowQueryInfo*>& _dvec );

	void depressed_set_windowQueryInfo_set(int sc_band, 	vector<GpuWindowQuery>& _windowQuerySet);
	void set_windowQueryInfo_set(int sc_band, vector<GpuWindowQuery>& _windowQuerySet);
	void depressed_set_windowQueryInfo_GpuQuery_newEntry(int groupQueryId, int slidingWindowId_idx,
			GpuWindowQuery& _gq_sw);
	void set_windowQueryInfo_GpuQuery_newEntry(int groupQueryId, int slidingWin_dataIdx,
			GpuWindowQuery& _gq_sw);
	void depressed_conf_windowQuery_onGpuManager(int sc_band);
	void depressed_conf_dev_windowQuery();
	void conf_dev_windowQuery();


	//for continous prediction
public:
	void disable_winQuery_ACDUpdate_forLBKeogh(){depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled = true;};
	void enable_winQuery_ACDUpdate_forLBKeogh(){depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled = false;};
	void depressed_update_ContQueryInfo_set(vector<vector<float> > gqi_data_set,vector<int> gqi_data_indicator);
	void update_ContQueryInfo_set(vector<vector<float> > gqi_data_set,vector<int> gqi_data_newData_indicator);
	void depressed_reset_TSGPUMananger_forGroupQuery();
	void reset_TSGPUMananger_forGroupQuery();
	void depressed_init_winQuery_ACDUpdate_forLBKeogh();
private:
	void depressed_update_ContQueryInfo(int groupQueryInfo_id,vector<float> gqi_data, int gqi_data_indicator);
	void update_ContQueryInfo(int groupQueryInfo_id,vector<float> gqi_data, int gqi_data_newData_indicator);
	void update_WindowQueryInfo(host_vector<WindowQueryInfo*> queryInfo_vec);

	/**
	 * depressed: replaced by labelReset_windowQuery_lowerBound() and dev_batch_reset_windowQuery_lowerBound();
	 */
	void depressed_reset_windowQuery_lowerBound(int windowQuery_info_id);


	void labelReset_windowQuery_lowerBound(int windowQuery_info_id);
	/**
	 * reset the bound as zero after the function labelReset_windowQuery_lowerBound(windowQuery_id) with batch method
	 */
	void dev_batch_reset_windowQuery_lowerBound();

	/**
	 * to re-calculate lower and uppper lower bound or effected window queries after new point add to the queries
	 */
	void dev_batch_reset_windowQueryInfo_entry_upperAndLowerBound();

	void update_windowQuery_entryAndTable(GpuWindowQuery& windowQuery);
	void update_windowQuery_entryAndTable(int winQueryId,int groupQueryId, int slidingWin_dataIdx);
	void update_windowQuery_entryAndTable(WindowQueryInfo* host_queryInfo, int d_query_info_id);
	void update_windowQueryInfo_entry_fromHostToDevice(WindowQueryInfo* host_queryInfo, WindowQueryInfo* device_queryInfo);

	void update_windowQueryInfo_upperAndLowerBound_fromHostToDevice(WindowQueryInfo* device_queryInfo,
				vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist);
	void update_windowQueryInfo_entry_upperAndLowerBound(int queryId, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist);
	void update_winQuery_LBKeoghAndTable(int new_window_start, int groupQueryInfo_id );


	void depressed_reset_winQuery_ACDUpdate_forLBKeogh();
	void reset_windowQuery_LBKeogh_updateCtrl();
	void depressed_update_winQuery_ACDUpdate_forLBKeogh_fromHostToDevice();

	int inline getWinQueryId(GroupQuery_info* gqi,int idx){
		return (idx+gqi->startWindowId)%windowNum_perGroup;
	}
	//get the location of winQuery in the given group query
	int inline getWinQueryLocOfGroupQuery(GroupQuery_info* gqi, int winQueryId){

		return (winQueryId-gqi->startWindowId+windowNum_perGroup)%windowNum_perGroup;

	}

	void inline increase_startwindowId(GroupQuery_info* gqi,int delta){
		gqi->startWindowId = (delta+gqi->startWindowId)%(windowNum_perGroup);
	};

private:
	//function with debug purpose
	void print_d_groupQuerySet_candidates_scan_byEndIdx(device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
	device_vector<int>& d_groupQuerySet_candidates_scan_endIdx);
	void print_d_groupQuerySet_candidates_scan_bySize(
			device_vector<CandidateEntry>& d_groupQuerySet_candidates_scan,
			device_vector<int>& d_groupQuerySet_candidates_scan_size);
	void print_d_groupQuerySet_candidates_scan_size( device_vector<int>& d_groupQuerySet_candidates_scan_size);
	void print_d_groupQuerySet_candidates_verified(int topk,device_vector<CandidateEntry>& d_groupQuerySet_candidates_verified,
		device_vector<int>& d_groupQuerySet_candidates_verified_size);
	void depressed_print_windowQuery_LowerBound();
	void print_windowQuery_lowerBound();
	void depressed_print_groupQuery_LowerBound();
	void print_groupQuery_LowerBound();
	void print_ACDUpdate_forLBKeogh();

//member variable

	/**
	 *
	 *
	 *
__global__ void compute_windowQuery_LowerBound(
	QueryFeatureEnt* windowQuery_feature,//input:
	QueryInfo** query_set, // input: remember the expansion position for each window query
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound and upper bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec
	);
	 *
	 *
	 */
public:
	GPUManager depressed_gpuManager;
	int sc_band;
	int windowNum_perGroup;//each group query has the same number of window queries
	int groupQuery_maxdimension;
	int groupQuery_item_num;//number of goup query dimensions, assumption: all group have the same number of different query dimensions

	int winDim;


	// query to keyword: each query associate with (query_spec_host.totalDimension) keywords
	device_vector<WindowQueryInfo*> d_windowQuery_info_set;
	host_vector<WindowQueryInfo*> h_windowQuery_info_set;
	int maxWindowFeatureNumber;
	bool enhancedLowerBound;
	int enhancedLowerBound_sel;//select the mode for enhancedLowerBound, 	enhancedLowerBound_sel: 0: use d2q, 1: use q2d, 2: use max(d2q,q2d)
	// result of lower bound for each window query
	device_vector<windowQuery_boundEntry> d_windowQuery_lowerBound;//default d2q
	device_vector<windowQuery_boundEntry> d_windowQuery_lowerBound_q2d;
	double sum_unfiltered_candidates;
	bool enable_sum_unfiltered_candidates;

	device_vector<int> d_windowQuery_lowerBound_endIdx;

	//this is to control update the LBKeogh for window query
	device_vector<bool> d_windowQuery_LBKeogh_updatedLabel;
	host_vector<bool> h_windowQuery_LBKeogh_updatedLabel;

	//this function is to update the value of ACD in count&ACD table when do continuous prediction,
	bool depressed_winQuery_ADCUpdate_forLBKeogh_masterCtrl;
	bool depressed_winQuery_ADCUpdate_forLBKeogh_isDisabled;
	device_vector<bool> depressed_d_winQuery_LBKeogh_ACDUpdate_subCtrlVec; //size: number of window queries
	host_vector<bool> depressed_h_winQuery_LBKeogh_ACDUpdate_subCtrlVec;
	device_vector<float> depressed_d_winQuery_LBKeogh_ACDUpdate_valueVec;//size: sc_band * number of window queries
	host_vector<float> depressed_h_winQuery_LBKeogh_ACDUpdate_valueVec;




	//gropu query part

	host_vector<GroupQuery_info*>   h_groupQuery_info_set;
	device_vector<GroupQuery_info*> d_groupQuery_info_set;
	device_vector<bool> depressed_d_groupQuery_unfinished;//default values are true
	device_vector<CandidateEntry> d_groupQuerySet_topkResults;
	device_vector<int> d_groupQuerySet_topkResults_size;//the true topk size may be smaller than the topk value, the true size is stored here
	device_vector<float> d_groupQuerySet_items_lowerBoundThreshold;//each group query has a lower bound threshold
	//lower bound for each group query, this is the partial sum of d_winQuery_result, which record the lower bound for all features
	device_vector<groupQuery_boundEntry> d_groupQuerySet_lowerBound;

	//different group queries may have different search extreme, i.e. [0~55]
	vector<int> depressed_groupQuery_upSearchExtreme;//the true max extreme value that the query may reach (i.e. 127), this is different from the lgical value such as 2^8
	vector<int> depressed_groupQuery_downSearchExtreme;//the true min extreme value that the query may reach (i.e. 0)

	//blade data part
	int ts_blade_num;
	device_vector<float> d_ts_data;//un initialized !!!, there should be multiple blades since multiple sensors
	device_vector<float> d_ts_data_u;// the LB_keogh upper bound for the time series data, compute it after configuring d_ts_data
	device_vector<float> d_ts_data_l;// the LB_keogh lower bound for the time series data, compute it after configuring d_ts_data
	device_vector<int> d_ts_data_blade_endIdx;//note: indicate the starting position for each blade in d_ts_data
	host_vector<int> h_ts_blade_len;

	//for profiling
	//vector<double> exec_time;
	//vector<string> exec_time_info;

	map<string,double> exec_timeInfo_set;
	map<string,int> exec_timeInfo_count_set;
	map<string,struct timespec> exec_timespec_set;
	map<string, cudaEvent_t> exec_cudaEvent_t_set;

};

#endif /* TSGPUMANAGER_H_ */
