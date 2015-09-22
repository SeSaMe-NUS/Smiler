/*
 * TSGPUFunctions.h
 *
 *  Created on: May 10, 2014
 *      Author: zhoujingbo
 */

#ifndef TSGPUFUNCTIONS_H_
#define TSGPUFUNCTIONS_H_
#include <stdio.h>
#include <unistd.h>

#include "../GPUKNN/UtlGPU.h"
#include "../GPUKNN/generalization.h"
#include "UtlTSProcess.h"


#define depressed_VERIFIED_LABEL_VALUE (FLT_MAX) //label value for verified
#define INITIAL_LABEL_VALUE ((float)INT_MAX-1) //label value for initial data, which means this features is impossible

class TSGPUFunctions {
public:
	TSGPUFunctions();
	virtual ~TSGPUFunctions();

};

//typedef struct GroupQuery_info{
class GroupQuery_info{


public:
	__host__ __device__ GroupQuery_info(int groupId, int blade_id, int startQueryId, int* groupQueryDimensions_vec,
			int groupQuery_item_number,float* gq_data){
		this->groupId = groupId;
		this->startWindowId = startQueryId;//the window id is counted in natural way
		this->blade_id = blade_id;
		this->item_num = groupQuery_item_number;

		this->item_dimensions_vec = new int[groupQuery_item_number];
		this->depressed_item_query_finished  = new bool [groupQuery_item_number];
		for(int i=0;i<groupQuery_item_number;i++){
			this->item_dimensions_vec[i] = groupQueryDimensions_vec[i];
			this->depressed_item_query_finished[i] = false;
		}

		data = new float[item_dimensions_vec[item_num-1]];
		for(int i=0;i<item_dimensions_vec[item_num-1];i++){
			data[i] = gq_data[i];
		}

	}

	__host__ __device__ GroupQuery_info(){
			this->groupId = -1;
			this->blade_id=-1;
			this->startWindowId = -1;
			item_dimensions_vec = NULL;
			depressed_item_query_finished = NULL;
			this->data=NULL;
			item_num = 0;

	}

	__host__ __device__ ~GroupQuery_info(){

		if(NULL != item_dimensions_vec)  {delete[] item_dimensions_vec; item_dimensions_vec = NULL;}
		if(NULL != depressed_item_query_finished)  {delete[] depressed_item_query_finished; depressed_item_query_finished = NULL;}
		if(NULL != data)	{delete[] data; data = NULL;}
	}

public:
	__host__ __device__ void depressed_setItemQueryFinished(int itemId){
		//int id = (itemId+startWindowId)%groupQuery_item_number;

		depressed_item_query_finished[itemId] = true;
	}


	__host__ __device__ bool depressed_isItemQueryFinished(int itemId){
		//int id = (itemId+startWindowId)%groupQuery_item_number;
		return depressed_item_query_finished[itemId];
	}

	//inclusive the start and  end
	__host__ __device__ bool depressed_areItemQueriesFished(int itemStart, int itemEnd){

		bool res = true;
		//int id;
		for(int i= itemStart;i<=itemEnd;i++){
		//	id = (i+ startWindowId)%groupQuery_item_number;
			res&=depressed_item_query_finished[i];
		}

		return res;
	}

	__host__ __device__ bool isGroupQueryFinished(){
		return depressed_areItemQueriesFished(0,item_num-1);
	}

	__host__ __device__ int getMaxQueryLen(){
		return item_dimensions_vec[item_num-1];
	}

	__host__ __device__ int getItemDimension(int item){
		return item_dimensions_vec[item];
	}

	__host__ __device__ void print(){

		printf("print GroupQuery_info=================================================================\n");
		printf("groud id =%d blade_id = %d \n",groupId, blade_id);
		printf("print item dimension vector:\n");

		for(int i=0;i<item_num;i++){
			printf(" item_dimensions_vec[%d]=%d   ", i, item_dimensions_vec[i]);
		}
		printf("\n");

		int queryLen = item_dimensions_vec[item_num -1];
		for(int i=0;i<queryLen;i++){
			printf(" data[%d] = %f",i,data[i]);
		}
		printf("\n");
		printf("end print GroupQuery_info==============================================================\n");
	}

	__host__ __device__ void depressed_reset_item_query_asUnfinished(){
		for(int i=0;i<item_num;i++){
			depressed_item_query_finished[i] = false;
		}
	}


public:
	int groupId;
	int blade_id;//which blade this group query belong to. For example ,there may be two sensors, the blade_id indicates which sensor's data this group query belong to.
	int startWindowId;


	/*for the same group query, we can set multiple dimensions, and align the tailed(latest) groupQueryDimensions data points as query
	 * i.e.for a whole query, we may have multiple queries with the latest data points
	 *
	 * note:1. The last one is the maximum dimension for this query
	 *      2. to count the dimension, load data right side first, from tail to head
	 */
	int* item_dimensions_vec;
	bool* depressed_item_query_finished;
	int item_num;//note: in all group queries, this variable should be equal to dev_groupQuery_dimensions_vec_len[1]

	float* data;//the longest query data, which should be equal groupQuery_dimensions_vec[groupQuery_dimensions_number - 1]

};





/* entries for Sum Table and Join Table, only record the lower bound */
typedef float windowQuery_boundEntry;




typedef float groupQuery_boundEntry;//note: if =  MAX, this boundEntry is verified. intialized as 0

struct ValueOfGroupQuery_boundEntry{
__host__ __device__ float valueOf(groupQuery_boundEntry data)
{
   		return data;
	}
};


typedef struct CandidateEntry{
	int feature_id;	// feature ID,
	float dist;			// lower bound


	__host__ __device__ CandidateEntry(){
		this->feature_id = -1;
		this->dist = 0;
	}

	__host__ __device__ CandidateEntry(int id, float dist){
		this->feature_id = id;
		this->dist = dist;

	}

	__host__ __device__ void print(){

		printf("feature id %d dist %f\n",feature_id,dist);
	}
} CandidateEntry;


struct ValueOfCandidateEntry {
	__host__ __device__ float valueOf(CandidateEntry data) {
		return data.dist;
	}
};


struct compare_CandidateEntry_dist {
    __host__ __device__
    bool operator()(CandidateEntry lhs, CandidateEntry rhs)
    {
      return lhs.dist < rhs.dist;
    }
};

struct LBKeogh_L2_distance{

	__host__ __device__ LBKeogh_L2_distance(){

	}

	__host__ __device__ float inline dist(float dim_data_value, float dim_keyword, float dim_lb_dist, float dim_ub_dist) {
			{
				float result;

				float diff = fabs(dim_data_value - dim_keyword);
				float bound = dim_data_value<=dim_keyword?dim_lb_dist:dim_ub_dist;

				result = (diff <= bound) ? 0 : (diff - bound) * (diff - bound);

				return result;
			}
		}

};


struct LBKeogh_L1_distance{

	__host__ __device__ LBKeogh_L1_distance(){

	}

	__host__ __device__ float inline dist(float dim_data_value, float dim_keyword, float dim_lb_dist, float dim_ub_dist) {
			{
				float result;

				float diff = fabs(dim_data_value - dim_keyword);
				float bound = dim_data_value<=dim_keyword?dim_lb_dist:dim_ub_dist;

				result = (diff <= bound) ? 0 : (diff - bound) ;
				return result;
			}
		}

};

struct LBKeogh_distance {

	__host__ __device__ LBKeogh_distance() {

	}

// distance function for values on the same dimension
	//dist(dim_data_value,dim_keyword,dim_lb_dist,dim_ub_dist,distFuncType)
	__host__ __device__ float inline dist(float dim_data_value, float dim_keyword, float dim_lb_dist, float dim_ub_dist, float distFuncType) {
		{
			float result;

			float diff = fabs(dim_data_value - dim_keyword);
			float bound = dim_data_value<=dim_keyword?dim_lb_dist:dim_ub_dist;

			if (distFuncType == 0) {
				result = (diff <= bound) ? 0 : 1; // the bound step function
				return result;
			} else if (distFuncType == 1) {

				result = (diff <= bound) ? 0 : (diff - bound); // the bound step function

			} else if (distFuncType == 2) {

				result = (diff <= bound) ? 0 : (diff - bound) * (diff - bound);

			} else {
				result = (diff <= bound) ? 0 : powf(diff - bound, distFuncType);
			}

			//result = (diff < bound) ? 0 : 1; // the bound step function, how to involve the bound
			//result = (diff == 0) ? 0 : result; // make sure the 0 case is correct
			//result *= dim_weight; // add dimension weight

			return result;
		}
	}

};



//__host__ __device__ float eu_flt( const float* Q, uint sq, const float* C, uint sc, uint cq_len);
/**
 * r:  Sakoe-Chiba Band
 * TODO:
 * 1. the data points of Q and C is from 1 to cq_len
 * Q[0]=C[0]=infinity
 * refer to paper "Accelerating dynamic time warping subsequence search with GPUs and FPGAs, ICDM, but the Algorithm in Table2 is wrong (or confused). This is the correct one
 *
 *
 * 2. Add Sakoe-Chiba Band to compute the DTW
 * refer to paper: Sakoe, Hiroaki, and Seibi Chiba. "Dynamic programming algorithm optimization for spoken word recognition."
 *  Acoustics, Speech and Signal Processing, IEEE Transactions on 26, no. 1 (1978): 43-49.
 *
 * 3. =======================================
 * corrected pseudo code  Algorithm:
 * mod = 2*r+1+1//one elements to store the maximum value, ether one is the element self, i.e. INF X X X X => X INF X X X
 *
 * for i=1 to mod-1:
 * 	   d(i,0) =  infinity
 *
 * d(0,1) = infinity;
 *
 * for j = 1 to n
 * 	   d((j-r-1)%mod,j%2) = infinity
 * 	   d((j+r)%mod,(j-1)%2) = infinity
 * 	   for i = j-r to j+r
 * 	       d(i%mod,j%2) = |C(j)-Q(i)| + min(d((i-1)%mod),j%2), d(i%mod, (j-1)%2), d((i-1)%mod, (j-1)%2)) //j - r <= i <= j + r
 *
 *
 * return d(n, n)
 *
 *
 *4. The two time of SC_band should be smaller than MAX_SCBAND=32, since if this sc_band is too large, the shared memory cannot handle the wrapping matrix and we need to
 *   spill the data into global memory
 */

__host__ __device__ float dtw_DP_SCBand_modulus_flt(const float* Q, uint q_len,const float* C, uint c_len, uint r);



/**
 * r:  Sakoe-Chiba Band
 * TODO:
 * 1. the data points of Q and C is from 1 to cq_len
 * Q[0]=C[0]=infinity
 * refer to paper "Accelerating dynamic time warping subsequence search with GPUs and FPGAs, ICDM, but the Algorithm in Table2 is wrong (or confused). This is the correct one
 *
 *
 * 2. Add Sakoe-Chiba Band to compute the DTW
 * refer to paper: Sakoe, Hiroaki, and Seibi Chiba. "Dynamic programming algorithm optimization for spoken word recognition."
 *  Acoustics, Speech and Signal Processing, IEEE Transactions on 26, no. 1 (1978): 43-49.
 *
 * 3. =======================================
 * pseudo code  Algorithm:
 * s = 0
 * for i=0 to m:
 * 	   d(i,s) =  infinity
 *
 * s = s XOR 1//Xor operation
 *
 * for j = 1 to n
 * 	   for i = j-r to j+r
 * 	       d(i,s) = |C(j)-Q(i)| + min(d(i-1),s), d(i, s XOR 1), d(i-1, s XOR 1)) //j - r <= i <= j + r
 * 	   s = s XOR 1
 *
 * return d(n,s XOR 1)
 *
 *
 *
 */
//template <class T>
__host__ __device__ float dtw_DP_SCBand(float* Q, uint q_len, float* C, uint c_len, uint r);

__host__ __device__ float eu_L1_flt( const float* Q, uint sq, const float* C, uint sc, uint cq_len);
__host__ __device__ float eu_L2_flt( const float* Q, uint sq, const float* C, uint sc, uint cq_len);


/**
 * the distance function with sc_band
 */

struct Dtw_SCBand_Func_modulus_flt{

	uint sc_band;

	__host__ __device__ Dtw_SCBand_Func_modulus_flt(uint sc_band){
		this->sc_band = sc_band;
	}

	__host__ __device__ float dist ( const float* Q, uint sq, const float* C, uint sc, uint cq_len){
		return  dtw_DP_SCBand_modulus_flt( Q+sq, cq_len,C+sc, cq_len, sc_band);//eu_L2_flt(Q, sq, C, sc, cq_len);  //
	}

};
//gpu operation function
void init_dev_groupQuery_constantMem( const int windwoNumber_perGroup,const int groupQuery_dimensions_number );
void init_dev_windowQuery_constantMem(const int winDim, const int maxWinFeatureID);





//GPU running code



template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__device__ void blk_get_eachWindowDimension_LowerBound_template(
		WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound,
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc
		);

__device__ void inline blk_get_eachDimension_LowerBound(WindowQueryInfo* queryInfo,int queryDimensionNum,
		float* _queryLowerBound);

template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__global__ void depressed_compute_windowQuery_LowerBound_template(
	QueryFeatureEnt* windowQuery_feature,//input:
	WindowQueryInfo** query_set, // input: remember the expansion position for each window query
	GpuIndexDimensionEntry* indexDimensionEntry_vec,//input record dimension infor, minDomain,maxDomain and bucketWidth
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound and upper bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec,
	KEYWORDMAP keywordMap,
	LASTPOSMAP lastPosMap,
	DISTFUNC distFunc
	);

//remove here
__global__ void depressed_compute_windowQuery_LowerBound(
	QueryFeatureEnt* windowQuery_feature,//input:
	WindowQueryInfo** query_set, // input: remember the expansion position for each window query
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound and upper bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec
	);

__global__ void compute_tsData_UpperLowerBound(
	float* d_ts_data,
	float* d_ts_data_u, float* d_ts_data_l,//output: record the lower and upper bound for ts_data
	int* d_ts_data_blade_endIdx,
	int sc_band
	);

template <class DISTFUNC>
__global__ void compute_windowQuery_lowerBound(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound,//output: record the lower bound and upper bound of each window query
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	int winDim, int sc_band,
	DISTFUNC distFunc);


template <class DISTFUNC>
__global__ void compute_windowQuery_enhancedLowerBound(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 float* d_ts_data_u,
	 float* d_ts_data_l,
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_d2q,//output: record the lower bound and upper bound of each window query, with d2q
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_q2d,//output: record the lower bound and upper bound of each window query, with q2d
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int winDim,  int sc_band,bool enhancedLowerBound,
	 DISTFUNC distFunc);

__global__ void prefixCount_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_result_vec, //input: record the lower bound  of each group query
		float* threshold, //input: threshold for each group query items
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_count_vec, //output: compute candidates per thread
		int* blk_count_vec, //output: compute candidates per block (kernel)
		GroupQuery_info** d_groupQuery_info_set //output:if the candidates are empty, set true for this group query (item).
		);

/**
 * edit zhou jingbo
 * TODO :  this function is to compute the lower bound for each possible feature with one query group. One query group compose with multiple slidig windows.
 * One set of disjoint sliding windows within query group forms one  equal class. We use one kernel to take one equal class
 *
 * note: 1. each block takes one equal class of one query
 **		 2.	number of blocks: total number of group queries multiply window dimension (the number of ECs is equal to window dimension
 **		 3. create the sliding windows in reverse order (from tailed to head) for group query.
 *
 */
__global__ void depressed_compute_groupQuery_LowerBound(
		windowQuery_boundEntry* windowQuery_result_vec, //input: record the lower bound and upper bound of each window query
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		bool* d_groupQuery_unfinished,//input: status of each group query
		groupQuery_boundEntry* groupQuery_result_vec //output: record the lower bound and upper bound of each group query
		);

__global__ void compute_groupQuery_lowerBound(
		windowQuery_boundEntry* windowQuery_bound_vec, //input: record the lower bound and upper bound of each window query
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		groupQuery_boundEntry* groupQuery_boundEntry_vec //output: record the lower bound and upper bound of each group query
		);

__global__ void compute_groupQuery_enhancedLowerBound(
		windowQuery_boundEntry* windowQuery_bound_vec_d2q, //input: record the lower bound and upper bound of each window query with d2q
		windowQuery_boundEntry* windowQuery_bound_vec_q2d, //input: record the lower bound and upper bound of each window query with q2d
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		groupQuery_boundEntry* groupQuery_boundEntry_vec, //output: record the lower bound and upper bound of each group query
		int enhancedLowerBound_sel
		);

__global__ void depressed_prefixCount_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_result_vec, //input: record the lower bound  of each group query
		float* threshold, //input: threshold for each group query items
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_count_vec, //output: compute candidates per thread
		int* blk_count_vec, //output: compute candidates per block (kernel)
		GroupQuery_info** d_groupQuery_info_set //output:if the candidates are empty, set true for this group query (item).
		);




/**edit zhou jingbo
 * TODO :  this function is to scan the groupQuery_result_vec and output the candidate whose lower bound is larger than the threshold of this group query
 *
 * note:1. one block take one query, i.e. block number is equal to groupQuery number
 *
 */

__global__ void depressed_output_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_item_boundEntry_vec, //input: record the lower bound of each group query
		float* threshold,//input: threshold for each group query
		GroupQuery_info** d_groupQuery_info_set, //input:check wheter this group query (item) is finished
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_result_vec_endIdx, //output: compute candidates per thread
		CandidateEntry* candidate_result//output: store the candidates for all group queries
	);

__global__ void output_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_item_boundEntry_vec, //input: record the lower bound of each group query
		float* threshold,//input: threshold for each group query
		GroupQuery_info** d_groupQuery_info_set, //input:check wheter this group query (item) is finished
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_result_vec_endIdx, //output: compute candidates per thread
		CandidateEntry* candidate_result//output: store the candidates for all group queries
	);

/**
 * reset as zero for lower bound for effected window queries after new point add to the queries
 * block number is equal to the maximum number of windows queries, i.e. length of windowQuery_LBKeogh_updatedLabel
 */
__global__ void kernel_batch_labelReset_windowQuery_lowerBound(
		bool* windowQuery_LBKeogh_updatedLabel,
		bool enhancedLowerBound,//whether to update the lowerbound
		int intVal,//set as 0
		windowQuery_boundEntry* windowQuery_lowerBound_d2q,
		windowQuery_boundEntry* windowQuery_lowerBound_q2d
	);


/**
//to re-calculate lower and uppper lower bound or effected window queries after new point add to the queries
//block number is equal to the maximum number of windows queries, i.e. length of windowQuery_LBKeogh_updatedLabel
* to replace the following functions:*
* ## to compute the lower and upper distance bound
*

void initQueryItem_LBKeogh(int sc_band,	float* queryData,
		int queryData_start, int queryData_len,
		int numOfDimensionToSearch,	float* keywords,
		vector<float>& upwardDistanceBound, vector<float>& downwardDistanceBound);
*
*
* ## to update to the GPU memeory
*
* void TSGPUManager::update_windowQueryInfo_entry_upperAndLowerBound(int queryId, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist);
 */


__global__ void kernel_batch_labelReset_windowQueryInfo_entry_upperAndLowerBound(
		bool* windowQuery_LBKeogh_updatedLabel,
		int sc_band,
		WindowQueryInfo** d_windowQuery_info_set,//output:
		GroupQuery_info** d_groupQuery_info_set//output
	);

/*
 * TODO:
 * 		verify the candidates
 * 		the number of blocks is the total queries, i.e. the number of group query multiply the number of item query with different dimensions per group
 */

__global__ void depressed_scan_verifyCandidates(
		const int sc_band,//input: Sakoe-Chiba Band
		const int candidates_number_fixed, //refer to note 1, if candidats_number_fixed == 0, the candidates number is flexible
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		const float* d_ts_data, //the time series blade, note: there may be multiple blades
		const int* d_ts_data_endIdx,//the endidx for each blades (i.e. the boundary of different blades)
		CandidateEntry* d_groupQuerySet_Candidates,//output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
		const int* d_groupQuerySet_Candidates_endIdx,//record the idx for each group query in d_groupQuerySet_Candidates
		groupQuery_boundEntry* groupQuery_boundEntry_vec //output: update the boundEntry after verification and set the verified as true
		);

template <class DIST>
__global__ void scan_verifyCandidates(
		const int sc_band, //input: Sakoe-Chiba Band
		const int candidates_number_fixed, //refer to note 1, if candidats_number_fixed == 0, the candidates number is flexible, else, this defines the fixed (maximum) possible number of candidates in d_groupQuerySet_candidates"
		GroupQuery_info** groupQuery_info_set, //input:group query
		const float* d_ts_data, //the time series blade, note: there may be multiple blades
		const int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
		CandidateEntry* groupQuerySet_candidates, //input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
		const int* groupQuerySet_candidates_idxInfo, //input:record the idx for each group query in d_groupQuerySet_Candidates
		//groupQuery_boundEntry* groupQuery_boundEntry_vec, //output: update the boundEntry after verification and set the verified as true
		DIST distFunc);


template <class DIST>
__global__ void scan_verifyCandidates_perThreadGenerated(
		const int sc_band, //input: Sakoe-Chiba Band
		GroupQuery_info** groupQuery_info_set, //input:group query
		const float* d_ts_data, //the time series blade, note: there may be multiple blades
		const int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
		CandidateEntry* groupQuerySet_candidates, //input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
		const int* groupQuerySet_candiates_thread_endIdx,
		const int threadPerBlock, const int mergeThreadInterval,
		//groupQuery_boundEntry* groupQuery_boundEntry_vec, //output: update the boundEntry after verification and set the verified as true
		DIST distFunc);

__global__ void select_candidates_mergeSort(
		const int topk,
		const int candidates_number_fixed,
		const int max_sharedMem_size,
		CandidateEntry* groupQuerySet_candidates,
		const int* d_groupQuerySet_Candidates_endIdx,//record the idx for each group query in d_groupQuerySet_Candidates
		CandidateEntry* groupQuerySet_candidates_kSelected,//output: result are stored here
		int * groupQuerySet_candidates_kSelected_size
		);

__global__ void select_candidates_random(
		const int candidates_num,
		CandidateEntry* groupQuerySet_candidates,
		const int* d_groupQuerySet_Candidates_endIdx, //record the idx for each group query in d_groupQuerySet_Candidates
		CandidateEntry* groupQuerySet_candidates_select, //output: result are stored here
		int* d_groupQuerySet_candidates_select_size//output: note: this is not endIdx, do not do inclusive sum, we only count the number of candidates
		);


__global__ void check_groupQuery_allItemsFinished(
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		bool* groupQuery_finished//output: the status of every group queries
		);

__global__ void maintain_topkCandidates_mergeSort(
	const int topk,
	CandidateEntry* groupQuerySet_topk_results,//input and output: there must be k result, no need for int to count the address
	CandidateEntry* groupQuerySet_candidates_kSelected,//input: result are stored here
	int * groupQuerySet_candidates_kSelected_size, //note this is size, not address, each query occupied k items size, but some be padded with NULL entry
	float* groupQuerySet_threshold
	);

#endif /* TSGPUFUNCTIONS_H_ */
