/*
 * TSGPUFunctions.cpp
 *
 *  Created on: May 10, 2014
 *      Author: zhoujingbo
 */

#include "TSGPUFunctions.h"
#include "../GPUKNN/generalization.h"
#include <cfloat>


TSGPUFunctions::TSGPUFunctions() {
	// TODO Auto-generated constructor stub

}

TSGPUFunctions::~TSGPUFunctions() {
	// TODO Auto-generated destructor stub
}

//----------------------------------for GPU constant memeory
__device__ __constant__ int dev_winDimension[1];//
__device__ __constant__ int dev_maxWindowNumber_perGroupQuery[1]; //assumption: all query have the same number of sliding windows
__device__ __constant__ int dev_item_num_perGroupQuery[1];//number of group query items, assumption: all group have the same number query items
__device__ __constant__ int dev_maxWinFeatureID[1]; //assumption: all data in different groups have the same number of maxWinFeatureID
//assumption: all data in different groups have the same number of maxWinFeatureID and the same length of winDim
//then dev_maxDataFeatureID = dev_maxWinFeatureID*dev_winDimension (since we use disjoint windows)
__device__ __constant__ int dev_maxDataFeatureID[1];



void init_dev_groupQuery_constantMem(const int windwoNumber_perGroup,const int items_number_perGroupQuery )
{
	printf ("# sliding windows of every group query: %d \n", windwoNumber_perGroup );
	printf ("# number items of every goup query: %d \n", items_number_perGroupQuery );


	HANDLE_ERROR( cudaMemcpyToSymbol( dev_maxWindowNumber_perGroupQuery, &(windwoNumber_perGroup), sizeof(int), 0, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_item_num_perGroupQuery, &(items_number_perGroupQuery ), sizeof(int), 0, cudaMemcpyHostToDevice ) );


}


void init_dev_windowQuery_constantMem(const int winDim,const int maxWinFeatureID){

	printf ("# window query dim: %d \n", winDim );
	printf ("# maxWinFeatureID: %d \n", maxWinFeatureID );
	printf ("# max data FeatureID: %d \n", winDim*maxWinFeatureID );

	int maxDataFeatureID = winDim*maxWinFeatureID;

	HANDLE_ERROR( cudaMemcpyToSymbol( dev_winDimension, &(winDim), sizeof(int), 0, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_maxWinFeatureID, &(maxWinFeatureID), sizeof(int), 0, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_maxDataFeatureID, &(maxDataFeatureID), sizeof(int), 0, cudaMemcpyHostToDevice ) );


}




//----------------for GPU code-------------------------------------------------

//#define MAX_VALUE_LABEL_TOLERANCE  0.0001
__host__ __device__ bool inline depressed_isVerified(groupQuery_boundEntry gbe){

	//return ((gbe > (VERIFIED_LABEL_VALUE-MAX_VALUE_LABEL_TOLERANCE)) && (gbe < (VERIFIED_LABEL_VALUE+MAX_VALUE_LABEL_TOLERANCE)))? true:false;
	return (gbe == depressed_VERIFIED_LABEL_VALUE)?true:false;
}


__host__ __device__ float eu_L2_flt( const float* Q, uint sq, const float* C, uint sc, uint cq_len){

	float d = 0;
	for(uint i=0;i<cq_len;i++){
		d+=(C[i+sc]-Q[i+sq])*(C[i+sc]-Q[i+sq]);
		//d+=abs(C[i+sc]-Q[i+sq]);
	}
	return d;
}

__host__ __device__ float eu_L1_flt( const float* Q, uint sq, const float* C, uint sc, uint cq_len){

	float d = 0;
	for(uint i=0;i<cq_len;i++){
		//d+=(C[i+sc]-Q[i+sq])*(C[i+sc]-Q[i+sq]);
		d+=abs(C[i+sc]-Q[i+sq]);
	}
	return d;
}

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
#define MAX_SCBAND 64//5004
struct WarpingMaxtrix_flt{

	float d[MAX_SCBAND][2];//rounding with the MAX_SCBAND

	uint mod;

	__host__ __device__ WarpingMaxtrix_flt(uint sc_band){

		this->mod = 2*sc_band+1+1;//one elements to store the maximum value, ether one is the element self

		for(int i=1;i< mod;i++){
			d[i][0]=(float)INT_MAX;
			d[i][1] = 0;
		}

		d[0][0] = 0;// d[0][0]=c(1)-q(0)
		d[0][1] = (float) INT_MAX;
	}

	__host__ __device__ float& operator() (int i, int j){
		return d[i%mod][j%2];
	}
};


__host__ __device__ float dtw_DP_SCBand_modulus_flt(const float* Q, uint q_len,const float* C, uint c_len, uint r){


	WarpingMaxtrix_flt d(r);

	float dist = 0;

	for(uint j=1;j<c_len+1;j++){

		uint start_i = j<r+1 ? 1 : (j-r);
		uint end_i = j+r>(q_len)? q_len:(j+r);

		d(start_i-1,j) = (float) INT_MAX;
		if(j+r<= q_len){
			d(j+r,j-1) = (float) INT_MAX;
		}

		for(uint i = start_i;i<=end_i;i++){

			dist = (C[j-1]-Q[i-1])*(C[j-1]-Q[i-1]);
			//dist = abs(C[j-1]-Q[i-1]);
			d(i,j) = dist + fminf(d(i-1,j),fminf(d(i,j-1),d(i-1,j-1)));
		}

	}

	dist = d(q_len,c_len);


	return dist;
}



__host__ __device__ float dtw_DP_SCBand(float* Q, uint q_len, float* C, uint c_len, uint r)
{

	uint s=0;
	float** d;
	d = new float*[q_len+1];//create matrix for wrap path
	for(uint i=0;i<q_len+1;i++){
		d[i] =new float[2];
		for(uint j=0;j<2;j++){
			d[i][j] = 0;
		}
	}


	d[0][0] = 0;// d[0][0]=c(1)-q(0)
	d[0][1] = INT_MAX;
	for(uint i=1;i<q_len+1;i++){
		d[i][s]=INT_MAX;//(C[1]-Q[i])*(C[1]-Q[i]);
	}

	s=1-s;
	uint s_xor;
	float dist = 0;

	for(uint j=1;j<c_len+1;j++){

		uint start_i = j<r+1 ? 1 : (j-r);
		uint end_i = j+r>(q_len)? q_len:(j+r);

		d[start_i-1][s] = INT_MAX;
		s_xor = 1-s;
		if(j+r<=(q_len)){
		d[j+r][s_xor] = INT_MAX;
		}

		for(uint i = start_i;i<=end_i;i++){

			dist = (C[j]-Q[i])*(C[j]-Q[i]);
			d[i][s] = dist + fminf(d[i-1][s],fminf(d[i][s_xor],d[i-1][s_xor]));
		}

		s = 1-s;
	}

	s_xor = 1-s;
	dist = d[q_len][s_xor];

	for(uint i=0;i<q_len+1;i++){
		delete[] d[i];
	}
	delete[] d;

	return dist;
};






/**
 * author: zhou jignbo
 * TODO:
 * compute the lower bound and upper bound for function output_result_bidrection_search_KernelPerQuery
 *
 * * Edit note:
 * 	1.add template to support bucketlized index whit bucket width > 1 (by jingbo)
 */
template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__device__ void blk_get_eachWindowDimension_LowerBound_template(
		WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound,
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc
		){
	__syncthreads();
	int round = (block_num_search_dim) / blockDim.x
			+ ((block_num_search_dim) % blockDim.x != 0);
	for (int i = 0; i < round; i++) {
		int idx = i * blockDim.x + threadIdx.x;
		if (idx < block_num_search_dim) {

			float data_dim_keyword = queryInfo->keyword[idx];
			float dim_dist_func_lpType = queryInfo->depressed_distanceFunc[idx];
			float dim_weight = queryInfo->depressed_dimWeight[idx];
			float data_dist_lowerBound = queryInfo->lowerBoundDist[idx];//in data space
			float data_dist_upperBound = queryInfo->upperBoundDist[idx];//in data space
			int index_dim_upperBoundSearch = queryInfo->depressed_upperBoundSearch[idx];// search bound when going up, in index space
			int index_dim_lowerBoundSearch = queryInfo->depressed_lowerBoundSearch[idx];// search boudn when going down, in index space
			int dim = queryInfo->depressed_searchDim[idx];
			GpuIndexDimensionEntry indexDimEntry = indexDimensionEntry_vec[dim];
			int index_dim_keyword = keywordMap.mapping(data_dim_keyword,indexDimEntry.bucketWidth);

			_queryLowerBound[idx] = 0.;
			int2 q_pos = queryInfo->depressed_lastPos[idx];
			// make sure the bound is correct when upward and downward search all reach the maximum, comment by jingbo
			// modified to the min compare to modified.cu which use max

			bool isReachMin = (q_pos.x >= index_dim_lowerBoundSearch);//(q_pos.x	== (query_dim_value - indexDimensionEntry_vec[dim]));
			bool isReachMax = (q_pos.y >= index_dim_upperBoundSearch);//(q_pos.y == (dev_maxDomainForAllDimension[dim] - query_dim_value));

			int index_reach_up = index_dim_keyword + q_pos.y;
			int index_reach_down = index_dim_keyword - q_pos.x;

			float data_reach_up_value = lastPosMap.map_indexToData_up(index_reach_up, indexDimEntry.bucketWidth);
			float data_reach_down_value = lastPosMap.map_indexToData_down(index_reach_down,indexDimEntry.bucketWidth);

			float queryLowerBound_reach_up = distFunc.dist(data_dim_keyword, data_reach_up_value, dim_dist_func_lpType,
					data_dist_upperBound, dim_weight);
			float queryLowerBound_reach_down = distFunc.dist(data_dim_keyword, data_reach_down_value, dim_dist_func_lpType,
					data_dist_lowerBound, dim_weight);



			if (isReachMin && isReachMax) {
				_queryLowerBound[idx] = fmaxf(queryLowerBound_reach_up,queryLowerBound_reach_down);
			} else if (isReachMin) {
				_queryLowerBound[idx] = queryLowerBound_reach_up;
			} else if (isReachMax) {
				_queryLowerBound[idx] = queryLowerBound_reach_down;
			} else {
				_queryLowerBound[idx] = fminf(queryLowerBound_reach_up,queryLowerBound_reach_down);
			}
		}
	}

	__syncthreads();
}

template __device__ void blk_get_eachWindowDimension_LowerBound_template<DataToIndex_keywordMap_bucketUnit,IndexToData_lastPosMap_bucketUnit, Lp_distance>(
		WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound,
		DataToIndex_keywordMap_bucketUnit intListmap,
		IndexToData_lastPosMap_bucketUnit lastPosMap,
		Lp_distance inLpDist
		);

template __device__ void blk_get_eachWindowDimension_LowerBound_template<DataToIndex_keywordMap_bucket,IndexToData_lastPosMap_bucket_inclusive, Lp_distance>(
		WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound,
		DataToIndex_keywordMap_bucket intListmap,
		IndexToData_lastPosMap_bucket_inclusive lastPosMap,
		Lp_distance inLpDist
		);

/**
 * author: zhou jignbo
 * compute the lower bound for each dimension
 */
__device__ void inline blk_get_eachDimension_LowerBound(WindowQueryInfo* queryInfo,int queryDimensionNum,
		float* _queryLowerBound){
	//	float* _queryUpperBound){

	int round = (queryDimensionNum) / blockDim.x
			+ ((queryDimensionNum) % blockDim.x != 0);
	for (int i = 0; i < round; i++) {
		int idx = i * blockDim.x + threadIdx.x;
		if (idx < queryDimensionNum) {
			//int dim = queryInfo->searchDim[idx];
			float dim_dist_func_lpType = queryInfo->depressed_distanceFunc[idx];
			float dim_weight = queryInfo->depressed_dimWeight[idx];
			float dist_lowerBound = queryInfo->lowerBoundDist[idx];
			float dist_uppperBound = queryInfo->upperBoundDist[idx];
			int dim_upperBoundSearch = queryInfo->depressed_upperBoundSearch[idx];// search bound when going up, add by jingbo
			int dim_lowerBoundSearch = queryInfo->depressed_lowerBoundSearch[idx];// search boudn when going down, add by jingbo

			int2 q_pos = queryInfo->depressed_lastPos[idx];
			// make sure the bound is correct when upward and downward search all reach the maximum, comment by jingbo
			// modified to the min compare to modified.cu which use max

			bool isReachMin = (q_pos.x >= dim_lowerBoundSearch);//(q_pos.x	== (query_dim_value - dev_minDomainForAllDimension[dim]));
			bool isReachMax = (q_pos.y >= dim_upperBoundSearch);//(q_pos.y == (dev_maxDomainForAllDimension[dim] - query_dim_value));

			float min_lb = (float) INT_MAX;
			float min_lb_bound = (float) INT_MAX;

			if (isReachMin && isReachMax) {
				//min_lb = max(q_pos.x, q_pos.y);
				min_lb = (q_pos.x-dist_lowerBound) >= (q_pos.y-dist_uppperBound)? q_pos.x:q_pos.y;
				min_lb_bound = (q_pos.x-dist_lowerBound) >= (q_pos.y-dist_uppperBound)? dist_lowerBound:dist_uppperBound;
			} else if (isReachMin) {
				min_lb = q_pos.y;
				min_lb_bound = dist_uppperBound;
			} else if (isReachMax) {
				min_lb = q_pos.x;
				min_lb_bound = dim_lowerBoundSearch;
			} else {
				//min_lb = min(q_pos.x, q_pos.y);
				min_lb = (q_pos.x-dist_lowerBound) <= (q_pos.y-dist_uppperBound)? q_pos.x:q_pos.y;
				min_lb_bound = (q_pos.x-dist_lowerBound) <= (q_pos.y-dist_uppperBound)? dist_lowerBound:dist_uppperBound;
			}

			float min_value = 0;
			_queryLowerBound[idx] = depressed_distance_func(min_lb, min_value,
					dim_dist_func_lpType, min_lb_bound, dim_weight);

		}
	}
	__syncthreads();
}


/**
 * TODO:
 * compute the new ACD based on cauchy-schwarz inequality
 */
__device__ float inline updateFunc_ACD(float acd, float* acdUpdate_vec, int sc_band, int count){
	float expansion=0;
	for(int i=0;i<count&&i<sc_band;i++){
		expansion+=(acdUpdate_vec[i]*acdUpdate_vec[i]);
	}

	//innerProduct_cauchy <= sqrtf(expansion*acd);
	float newACD = (acd+expansion-2*sqrtf(expansion*acd));
	newACD = (newACD>=0) ? newACD:0;
	return newACD;

}




/**
 * edit zhou jingbo
 * TODO :  this function is to compute the lower bound for every  tableEntry based on the bi-direction expansion one kernel takes one count&ACD table
 *
 * parameter:
 * ACDUpdate: whether consider the continuous prediction to update the ACD of count&ACD table, note that for one continous prediction, we only need to update
 * the count&ACD table once.
 *
 * note: 1. define window dimension, and also allocate the shared memory
 *       2. each block takes one query
 *
 *	number of blocks: total number of window queries
 *	shared memory size: 3*dev_winDimension*sizeof(float)
 *
 *	edit note:
 *	1. change to templated method to computer lower bound of bucketedlized inverted list (2014.06.11)
 *
 */
template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__global__ void depressed_compute_windowQuery_LowerBound_template(
	QueryFeatureEnt* windowQuery_feature,//input:
	WindowQueryInfo** query_set, // input: remember the expansion position for each window query
	GpuIndexDimensionEntry* indexDimensionEntry_vec,//input record dimension infor, minDomain,maxDomain and bucketWidth
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec,
	KEYWORDMAP keywordMap,
	LASTPOSMAP lastPosMap,
	DISTFUNC distFunc
	){

	WindowQueryInfo* queryInfo = query_set[blockIdx.x];//get the window query

	// dynamic allocate shared memory. specify outside in kernel calls, should be 2 times of the window dimensions.
	extern __shared__ float sharedMem_dataArray[];
	float* queryLowerBound = sharedMem_dataArray;
	//float* queryUpperBound = &queryBoundMem[dev_winDimension[0]];
	float* temp_shared = &sharedMem_dataArray[dev_winDimension[0]];
	float* acdUpdate_vec =  &sharedMem_dataArray[2*dev_winDimension[0]];

	//got the lower bound for each dimension
	blk_get_eachWindowDimension_LowerBound_template(
			queryInfo,dev_winDimension[0], indexDimensionEntry_vec,
			queryLowerBound,
			keywordMap,lastPosMap,distFunc
	);




	// sort the lower and up bound
	blk_sort_inSharedMemory(queryLowerBound, temp_shared, dev_winDimension[0]);
	float lower_bound_sum = blk_sum_sharedMemory(temp_shared,dev_winDimension[0]);



	bool wq_updateControl = false;
	if(ACDUpdate_control){
	// sort the lower and up bound
		wq_updateControl = windowQuery_ACDUpdate_control[blockIdx.x];
		if(wq_updateControl){
			if(threadIdx.x < sc_band){
				acdUpdate_vec[threadIdx.x] = windowQuery_ACD_update_valueVec[blockIdx.x*sc_band+threadIdx.x];
			}
			__syncthreads();
			blk_sort_inSharedMemory(acdUpdate_vec, temp_shared, sc_band);
		}
	}


	int block_start = blockIdx.x * (*dev_maxWinFeatureID);//start of per window query
	int block_end = block_start + (*dev_maxWinFeatureID);//end
	int round = (*dev_maxWinFeatureID) / blockDim.x
			+ ((*dev_maxWinFeatureID) % blockDim.x != 0);
	__syncthreads();

	for(int i = 0; i < round; i++){

		int idx = block_start + i * blockDim.x + threadIdx.x;
		if(idx < block_end)
		{
			QueryFeatureEnt wq_item =  windowQuery_feature[idx];
			int count =wq_item.count;

			windowQuery_boundEntry new_result;
		//	new_result.feature_id = i * blockDim.x + threadIdx.x;

			// DL(f) = DL(us) - sum(max count number of dt) + ACD(f)
			new_result = lower_bound_sum;
			for (int m = 0; m < count; m++) {
				new_result -= queryLowerBound[m];
			}

			if(ACDUpdate_control){
				wq_item.ACD = updateFunc_ACD(wq_item.ACD, acdUpdate_vec, sc_band, count);
				windowQuery_feature[idx] = wq_item;//write back to global memory
			}

			new_result += wq_item.ACD;

			windowQuery_result_vec[idx] = new_result;

		}
	}
}

template __global__ void depressed_compute_windowQuery_LowerBound_template<DataToIndex_keywordMap_bucketUnit,IndexToData_lastPosMap_bucketUnit, Lp_distance>(
	QueryFeatureEnt* windowQuery_feature,//input:
	WindowQueryInfo** query_set, // input: remember the expansion position for each window query
	GpuIndexDimensionEntry* indexDimensionEntry_vec,//input record dimension infor, minDomain,maxDomain and bucketWidth
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound and upper bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec,
	DataToIndex_keywordMap_bucketUnit intListmap,
	IndexToData_lastPosMap_bucketUnit lastPosMap,
	Lp_distance inLpDist
	);

template __global__ void depressed_compute_windowQuery_LowerBound_template<DataToIndex_keywordMap_bucket,IndexToData_lastPosMap_bucket_inclusive, Lp_distance>(
	QueryFeatureEnt* windowQuery_feature,//input:
	WindowQueryInfo** query_set, // input: remember the expansion position for each window query
	GpuIndexDimensionEntry* indexDimensionEntry_vec,//input record dimension infor, minDomain,maxDomain and bucketWidth
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound and upper bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec,
	DataToIndex_keywordMap_bucket intListmap,
	IndexToData_lastPosMap_bucket_inclusive lastPosMap,
	Lp_distance inLpDist
	);

/**
 * edit zhou jingbo
 * TODO :  this function is to compute the lower bound for every  tableEntry based on the bi-direction expansion one kernel takes one count&ACD table
 *
 * parameter:
 * ACDUpdate: whether consider the continuous prediction to update the ACD of count&ACD table, note that for one continous prediction, we only need to update
 * the count&ACD table once.
 *
 * note: 1. define window dimension, and also allocate the shared memory
 *       2. each block takes one query
 *
 *	number of blocks: total number of window queries
 *	shared memory size: 3*dev_winDimension*sizeof(float)
 *
 */
__global__ void depressed_compute_windowQuery_LowerBound(
	QueryFeatureEnt* windowQuery_feature,//input:
	WindowQueryInfo** query_set, // input: remember the expansion position for each window query
	windowQuery_boundEntry* windowQuery_result_vec,//output: record the lower bound and upper bound of each window query
	bool ACDUpdate_control,
	int sc_band,//to index the interval of every window query
	bool* windowQuery_ACDUpdate_control,
	float* windowQuery_ACD_update_valueVec
	){

	WindowQueryInfo* queryInfo = query_set[blockIdx.x];//get the window query

	// dynamic allocate shared memory. specify outside in kernel calls, should be 2 times of the window dimensions.
	extern __shared__ float sharedMem_dataArray[];
	float* queryLowerBound = sharedMem_dataArray;
	//float* queryUpperBound = &queryBoundMem[dev_winDimension[0]];
	float* temp_shared = &sharedMem_dataArray[dev_winDimension[0]];
	float* acdUpdate_vec =  &sharedMem_dataArray[2*dev_winDimension[0]];

	//got the lower bound for each dimension
	blk_get_eachDimension_LowerBound(queryInfo,dev_winDimension[0], queryLowerBound);//,queryUpperBound);


	// sort the lower and up bound
	blk_sort_inSharedMemory(queryLowerBound, temp_shared, dev_winDimension[0]);
	float lower_bound_sum = blk_sum_sharedMemory(temp_shared,dev_winDimension[0]);

	bool wq_updateControl = false;
	if(ACDUpdate_control){
	// sort the lower and up bound
		wq_updateControl = windowQuery_ACDUpdate_control[blockIdx.x];
		if(wq_updateControl){
			if(threadIdx.x < sc_band){
				acdUpdate_vec[threadIdx.x] = windowQuery_ACD_update_valueVec[blockIdx.x*sc_band+threadIdx.x];
			}
			__syncthreads();
			blk_sort_inSharedMemory(acdUpdate_vec, temp_shared, sc_band);
		}
	}


	int block_start = blockIdx.x * (*dev_maxWinFeatureID);//start of per window query
	int block_end = block_start + (*dev_maxWinFeatureID);//end
	int round = (*dev_maxWinFeatureID) / blockDim.x
			+ ((*dev_maxWinFeatureID) % blockDim.x != 0);
	__syncthreads();

	for(int i = 0; i < round; i++){

		int idx = block_start + i * blockDim.x + threadIdx.x;
		if(idx < block_end)
		{
			QueryFeatureEnt wq_item =  windowQuery_feature[idx];
			int count =wq_item.count;

			windowQuery_boundEntry new_result;
		//	new_result.feature_id = i * blockDim.x + threadIdx.x;

			// DL(f) = DL(us) - sum(max count number of dt) + ACD(f)
			new_result = lower_bound_sum;
			for (int m = 0; m < count; m++) {
				new_result -= queryLowerBound[m];
			}

			if(ACDUpdate_control){
				wq_item.ACD = updateFunc_ACD(wq_item.ACD, acdUpdate_vec, sc_band, count);
				windowQuery_feature[idx] = wq_item;//write back to global memory
			}

			new_result += wq_item.ACD;

			windowQuery_result_vec[idx] = new_result;
		}
	}
}

/**
//maxWinFeatureID
//block number is equal to the maximum number of windows queries, i.e. length of windowQuery_LBKeogh_updatedLabel
* to replace the following function
void TSGPUManager::reset_windowQuery_lowerBound(int windowQuery_info_id){

	int wqi_start = windowQuery_info_id*this->getMaxWindowFeatureNumber();
	int wqi_end = wqi_start+this->getMaxWindowFeatureNumber();

	this->h_windowQuery_LBKeogh_updatedLabel[windowQuery_info_id]=true;
	thrust::fill(this->d_windowQuery_lowerBound.begin()+wqi_start,this->d_windowQuery_lowerBound.begin()+wqi_end, 0);//clear the Count&ACD table of this window query//
	if(enhancedLowerBound){
		thrust::fill(this->d_windowQuery_lowerBound_q2d.begin()+wqi_start,this->d_windowQuery_lowerBound_q2d.begin()+wqi_end, 0);//clear the Count&ACD table of this window query//
	}
}
 */
__global__ void kernel_batch_labelReset_windowQuery_lowerBound(
		bool* windowQuery_LBKeogh_updatedLabel,
		bool enhancedLowerBound,//whether to update the lower bound of q2d
		int intVal,//set as 0
		windowQuery_boundEntry* windowQuery_lowerBound_d2q,
		windowQuery_boundEntry* windowQuery_lowerBound_q2d
	){
	int bid = blockIdx.x;
	if(!windowQuery_LBKeogh_updatedLabel[bid]) return;



	int tid = threadIdx.x;


	int round = dev_maxWinFeatureID[0]/blockDim.x+1;
	int wqi_start=bid*dev_maxWinFeatureID[0];

	for(int i=0;i<round;i++){
		int offset = i*blockDim.x+tid;
		if(offset<dev_maxWinFeatureID[0]){
			windowQuery_lowerBound_d2q[wqi_start+offset]=intVal;

			if(enhancedLowerBound){
				windowQuery_lowerBound_q2d[wqi_start+offset]=intVal;
			}
		}
	}

	return;
}


//get the location of winQuery in the given group query
__device__ int getWinQueryLocOfGroupQuery(GroupQuery_info& gqi, int winQueryId){

		return (winQueryId-gqi.startWindowId+dev_maxWindowNumber_perGroupQuery[0])%dev_maxWindowNumber_perGroupQuery[0];

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


__global__ void kernel_batch_labelReset_windowQueryInfo_entry_upperAndLowerBound(
		bool* windowQuery_LBKeogh_updatedLabel,
		int sc_band,
		WindowQueryInfo** d_windowQuery_info_set,//output:
		GroupQuery_info** d_groupQuery_info_set//output
	){
	int bid = blockIdx.x;
	if(!windowQuery_LBKeogh_updatedLabel[bid]) return;

	int gid = bid/dev_maxWindowNumber_perGroupQuery[0];//group query id
	int wqid =bid;// window query id
	int locStart=getWinQueryLocOfGroupQuery((*d_groupQuery_info_set[gid]),wqid);//starting loc of this window in the group query


	int round = dev_winDimension[0]/blockDim.x+1;

	for(int ri=0;ri<round;ri++){

		int wi=ri*blockDim.x+threadIdx.x;//for every dimension of the window query
		if(wi<dev_winDimension[0]){
			int s = (locStart + wi - sc_band <= 0) ?
						0 : (locStart + wi - sc_band);
			int e = (locStart + wi + sc_band >= (d_groupQuery_info_set[gid]->getMaxQueryLen() - 1)) ?
						(d_groupQuery_info_set[gid]->getMaxQueryLen() - 1) : (locStart + wi + sc_band);

			float up = -(float) INT_MAX; //find maximum value within [i-r,i+r]
			float down = (float) INT_MAX; //find minimum value within [i-r,i+r]

			for(int j=s;j<=e;j++){
				float gdv=d_groupQuery_info_set[gid]->data[j];
				if (up < gdv) {
					up = gdv;
				}
				if (down > gdv){
					down = gdv;
				}
			}

			up = up - d_windowQuery_info_set[wqid]->keyword[wi];
			down = d_windowQuery_info_set[wqid]->keyword[wi] - down;

			d_windowQuery_info_set[wqid]->upperBoundDist[wi] = up;
			d_windowQuery_info_set[wqid]->lowerBoundDist[wi] = down;
		}
	}

	return;
}
/**
 * TODO:
 * compute the upper and lower bound for each data blade
 * use one kernel to compute one blade, however, this can be more parallized
 */
__global__ void compute_tsData_UpperLowerBound(
	float* d_ts_data,
	float* d_ts_data_u, float* d_ts_data_l,//output: record the lower and upper bound for ts_data
	int* d_ts_data_blade_endIdx,
	int sc_band
	){

	int blade_id = blockIdx.x;
	int blade_start_idx = (blade_id == 0)? 0:d_ts_data_blade_endIdx[blade_id-1];
	int blade_end_idx = d_ts_data_blade_endIdx[blade_id];
	int blade_len = blade_end_idx - blade_start_idx;

	int round = (blade_len)/blockDim.x + ((blade_len)%blockDim.x!=0);

	for(int i=0;i<round;i++){
		int idx = i*blockDim.x + threadIdx.x;

		if(idx<blade_len){

			int sidx = (idx-sc_band)>=0? (idx-sc_band):0;
			int eidx = (idx+sc_band <= (blade_len-1)) ? (idx+sc_band):(blade_len-1);
			float dim_data=d_ts_data[blade_start_idx+idx];
			float l =(float)INT_MAX;
			float u = -(float)INT_MAX;

			for(int j=sidx;j<=eidx;j++){
				float de=d_ts_data[blade_start_idx+j];
				if(de<l) {l=de;}
				if(de>u){u=de;}
			}
			d_ts_data_u[blade_start_idx+idx] = u-dim_data;
			d_ts_data_l[blade_start_idx+idx] = dim_data-l;

		}

	}
}



/**
 * TODO:
 * compute the lower bound for each window query
 * note: use one kernel to take one dimension.
 */
template <class DISTFUNC>
__global__ void compute_windowQuery_lowerBound(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound,//output: record the lower bound and upper bound of each window query
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int winDim,  int sc_band,
	 DISTFUNC distFunc)
{
	int wq_id  =  blockIdx.x/winDim;
	bool updatedLabel = d_windowQuery_LBKeogh_updatedLabel[wq_id];

	if(!updatedLabel) return;

	int dimId= blockIdx.x%winDim; //mapped to different dimension index of A QUERY (note that: this dimId is to index the query dimension, i.e. by query->searchDim[dimId] to locate the true dimensions
	 WindowQueryInfo* wqi = windowQueryInfo_set[wq_id];

	float dim_lb_dist = wqi->lowerBoundDist[dimId];
	float dim_ub_dist = wqi->upperBoundDist[dimId];
	//float distFuncType = wqi->depressed_distFuncType;
	float dim_keyword = wqi->keyword[dimId];

	int blade_id =wqi->blade_id;//

	int blade_start_idx = (blade_id == 0)? 0:d_ts_data_blade_endIdx[blade_id-1];
	int blade_end_idx = d_ts_data_blade_endIdx[blade_id];
	int blade_len = blade_end_idx - blade_start_idx;
	int disjoint_win_num = blade_len/winDim;


	int jump  = dimId*winDim;//jump offset to avoid the atomic write

	int query_round = (disjoint_win_num)/blockDim.x + ((disjoint_win_num)%blockDim.x!=0);

	int wqr_start_idx = (wq_id==0)?0:d_windowQuery_lowerBound_endIdx[wq_id-1];

//
	for(int i=0;i<query_round;i++){
		int winIdx = i*blockDim.x+threadIdx.x;
		if(winIdx<disjoint_win_num){
			int winJumpIdx=(winIdx+jump)%disjoint_win_num;
			int idx = winJumpIdx*winDim+dimId+blade_start_idx;

			float dim_data_value = d_ts_data[idx];
			float dist = distFunc.dist(dim_data_value,dim_keyword,dim_lb_dist,dim_ub_dist);

			atomicAdd(&(_d_windowQuery_lowerBound[wqr_start_idx+winJumpIdx]), dist);

		}
	}
}


template __global__ void compute_windowQuery_lowerBound<LBKeogh_L2_distance>(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound,//output: record the lower bound and upper bound of each window query
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int sc_band,  int winDim,
	 LBKeogh_L2_distance distFunc);

template __global__ void compute_windowQuery_lowerBound<LBKeogh_L1_distance>(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound,//output: record the lower bound and upper bound of each window query
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int sc_band,  int winDim,
	 LBKeogh_L1_distance distFunc);




/**
 * TODO:
 * compute the lower bound for each window query with enchanced lower bound of LB_keogh
 * note: use one kernel to take one dimension.
 */
template <class DISTFUNC>
__global__ void compute_windowQuery_enhancedLowerBound(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 float* d_ts_data_u,
	 float* d_ts_data_l,
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_d2q,//output: record the lower bound and upper bound of each window query, d2q
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_q2d,//output: record the lower bound and upper bound of each window query, with q2d
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int winDim,  int sc_band, bool enhancedLowerBound,
	 DISTFUNC distFunc)
{
	int wq_id  =  blockIdx.x/winDim;
	bool updatedLabel = d_windowQuery_LBKeogh_updatedLabel[wq_id];

	if(!updatedLabel) return;

	int dimId= blockIdx.x%winDim; //mapped to different dimension index of A QUERY (note that: this dimId is to index the query dimension, i.e. by query->searchDim[dimId] to locate the true dimensions
	 WindowQueryInfo* wqi = windowQueryInfo_set[wq_id];

	float dim_lb_dist = wqi->lowerBoundDist[dimId];
	float dim_ub_dist = wqi->upperBoundDist[dimId];
	//float distFuncType = wqi->depressed_distFuncType;
	float dim_keyword = wqi->keyword[dimId];

	int blade_id =wqi->blade_id;//

	int blade_start_idx = (blade_id == 0)? 0:d_ts_data_blade_endIdx[blade_id-1];
	int blade_end_idx = d_ts_data_blade_endIdx[blade_id];
	int blade_len = blade_end_idx - blade_start_idx;
	int disjoint_win_num = blade_len/winDim;


	int jump  = dimId*winDim;//jump offset to avoid the atomic write

	int query_round = (disjoint_win_num)/blockDim.x + ((disjoint_win_num)%blockDim.x!=0);

	int wqr_start_idx = (wq_id==0)?0:d_windowQuery_lowerBound_endIdx[wq_id-1];
	int wqr_endIdx = d_windowQuery_lowerBound_endIdx[wq_id];

	if((wqr_endIdx-wqr_start_idx)!=disjoint_win_num){
		printf("with debug purpose error: disjoint_win_num=%d blade_len=%d",wqr_endIdx-wqr_start_idx,blade_len);
	}
//
	for(int i=0;i<query_round;i++){
		int winIdx = i*blockDim.x+threadIdx.x;
		if(winIdx<disjoint_win_num){
			int winJumpIdx=(winIdx+jump)%disjoint_win_num;
			int idx = winJumpIdx*winDim+dimId;

			//comput the distance from data to query
			float dim_data_value = d_ts_data[idx+blade_start_idx];
			float d2q_dist = distFunc.dist(dim_data_value,dim_keyword,dim_lb_dist,dim_ub_dist);
			atomicAdd(&(_d_windowQuery_lowerBound_d2q[wqr_start_idx+winJumpIdx]), d2q_dist);

			if(enhancedLowerBound){
				float u = d_ts_data_u[idx+blade_start_idx];
				float l = d_ts_data_l[idx+blade_start_idx];
				//float de=d_ts_data[idx+blade_start_idx];
				float q2d_dist = distFunc.dist(dim_keyword,dim_data_value,l,u);
				atomicAdd(&(_d_windowQuery_lowerBound_q2d[wqr_start_idx+winJumpIdx]), q2d_dist);
			}


		}

	}
}



template __global__ void compute_windowQuery_enhancedLowerBound<LBKeogh_L2_distance>(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 float* d_ts_data_u,
	 float* d_ts_data_l,
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_d2q,//output: record the lower bound and upper bound of each window query, d2q
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_q2d,//output: record the lower bound and upper bound of each window query, with q2d
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int sc_band,  int winDim, bool enhancedLowerBound,
	 LBKeogh_L2_distance distFunc);

template __global__ void compute_windowQuery_enhancedLowerBound<LBKeogh_L1_distance>(
	 float* d_ts_data, //the time series blade, note: there may be multiple blades
	 float* d_ts_data_u,
	 float* d_ts_data_l,
	 int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
	 WindowQueryInfo** windowQueryInfo_set, // input: records each window query
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_d2q,//output: record the lower bound and upper bound of each window query, d2q
	 windowQuery_boundEntry* _d_windowQuery_lowerBound_q2d,//output: record the lower bound and upper bound of each window query, with q2d
	 int* d_windowQuery_lowerBound_endIdx, //input: record the end idx for each windowQuery
	 bool* d_windowQuery_LBKeogh_updatedLabel,//input: label whether this lowerbound should be re-computed
	 int sc_band,  int winDim, bool enhancedLowerBound,
	 LBKeogh_L1_distance distFunc);


//auxiliary function for compute_groupQuery_LowerBound()

__device__ bool isItemQueryFinished(uint* item_query_finished, int itemId) {

	return item_query_finished[itemId];
}

//inclusive the start and  end
__device__ bool areItemQueriesFished(uint* item_query_finished, int itemStart, int itemEnd) {

	bool res = true;
	for (int i = itemStart; i <= itemEnd; i++) {
		res &= item_query_finished[i];
	}

	return res;
}

__device__ bool isGroupQueryFinished(uint* item_query_finished) {
	return areItemQueriesFished(item_query_finished, 0, dev_item_num_perGroupQuery[0] - 1);
}


__device__ int check_groupQuery_maxUnfinishedItem(uint* item_query_finished) {

	int res;

	for (int i = 0; i < dev_item_num_perGroupQuery[0]; i++) {

		if(!item_query_finished[i]) res = i;
	}

	return res;

}

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
		windowQuery_boundEntry* windowQuery_bound_vec, //input: record the lower bound and upper bound of each window query
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		bool* d_groupQuery_unfinished,
		groupQuery_boundEntry* groupQuery_boundEntry_vec //output: record the lower bound and upper bound of each group query
		) {

	//each block takes one equal class of one item query,  (the number of ECs is equal to window dimension)

	int group_qid = blockIdx.x / (dev_winDimension[0]); //each equal class (EC) is corresponding to multiple qury item in one group query
	bool groupQueryUnfinished = d_groupQuery_unfinished[group_qid];
	if (!groupQueryUnfinished)
		return;

	//one kernel takes one Equal Classes,the number of ECs per group is equal to window dimension
	//note: we compute the EC by the last sliding windows, i.e. the time series points with large time are within the first windows of ECs.
	int ec_id = blockIdx.x % dev_winDimension[0];
	GroupQuery_info* gquery = d_groupQuery_info_set[group_qid];

	//start of per window query. The number of candidate features for each group queries is number of windows features X window dims
	//Each EC process different segments of all the candidates
	//each group query has multiple different dimensions
	int groupQuery_boundEntry_vec_start = group_qid * dev_maxDataFeatureID[0] * (dev_item_num_perGroupQuery[0]);	//starting position to store result, jumping with window dimension
	int windowQuery_boundEntry_vec_start = group_qid * dev_maxWindowNumber_perGroupQuery[0] * (dev_maxWinFeatureID[0]);// //starting position to access lower bound for each EC

	//load the number of dimension into shared memeory

	extern __shared__ uint gq_sm[];

	uint* gq_item_dimesions = gq_sm;
	uint* gq_Item_finished = &gq_sm[dev_item_num_perGroupQuery[0]];

	if (threadIdx.x < dev_item_num_perGroupQuery[0]) {
		gq_item_dimesions[threadIdx.x] =
				gquery->item_dimensions_vec[threadIdx.x];
		gq_Item_finished[threadIdx.x] = gquery->depressed_item_query_finished[threadIdx.x];

	}
	__syncthreads();

	uint max_unfinished_ItemId = check_groupQuery_maxUnfinishedItem(
			gq_Item_finished);

	int gq_maxDimension = gq_item_dimesions[gquery->item_num - 1];

	// |------------------------------------------------------------|
	// |<--------------------------gq_maxDimension----------------->|
	// ||-----| |---------| |----------------| |--------------| |---|
	//                                                           EC starts from right
	//uint ec_index_leftBoundary = (gq_maxDimension - ec_id) / dev_winDimension[0]
	//		- (0 == ((gq_maxDimension - ec_id) % dev_winDimension[0]));

	int round = (*dev_maxWinFeatureID) / blockDim.x
			+ ((*dev_maxWinFeatureID) % blockDim.x != 0);//scan every elements, one element corresponding to multiple window queries within one EC and for different dimensions
	for (int i = 0; i < round; i++) {

		uint ec_index = i * blockDim.x + threadIdx.x;// + ec_index_leftBoundary; //within all time series features,stop to exceed boundary

		if (ec_index < dev_maxWinFeatureID[0] - (ec_id != 0)) //not exceed boundary
		{

			//compute lower bound for feature id for all window queries within one EC
			int gq_itemNum_count = 0;
			int gq_winNum_count = 0;
			groupQuery_boundEntry gq_item_BE = 0;

			//int EC_slidingWindow_Id_start = (gquery->startWindowId + ec_id)
			//		% dev_maxWindowNumber_perGroupQuery[0]; //loop with module of dev_maxWindowNumber_perGroup[0]

			int EC_slidingWindow_id_end = (gquery->startWindowId+dev_maxWindowNumber_perGroupQuery[0]-1)%dev_maxWindowNumber_perGroupQuery[0];//the last sliding window id of this group query
			int EC_sw_reverse_id_start = (EC_slidingWindow_id_end - ec_id)%dev_maxWindowNumber_perGroupQuery[0];

			//gq_j is the length of extended query
			for (uint gq_j = ec_id + dev_winDimension[0];//lenght of the extened query inclusive the ec offset part
					gq_j <= gq_maxDimension	&& gq_itemNum_count <= max_unfinished_ItemId && ec_index>= (gq_winNum_count+ (ec_id!=0));
					gq_j +=	dev_winDimension[0]) {

				int disWin_index = windowQuery_boundEntry_vec_start
						+ EC_sw_reverse_id_start * dev_maxWinFeatureID[0]
						+ ec_index - gq_winNum_count;
				gq_item_BE += windowQuery_bound_vec[disWin_index];

				if ((gq_j + dev_winDimension[0] > gq_item_dimesions[gq_itemNum_count])
						&& gq_j <= gq_item_dimesions[gq_itemNum_count]) {
					//TODO:here
					bool itemQueryFinished = isItemQueryFinished(gq_Item_finished, gq_itemNum_count);
					if (!itemQueryFinished) {

						int featureId = (ec_index - gq_winNum_count) * dev_winDimension[0]
								- ((dev_winDimension[0] - ec_id) % dev_winDimension[0]);// ec_index>= (gq_winNum_count+ (ec_id!=0) is to avoid exceeding boundary

						int featureId_glboalIdx = groupQuery_boundEntry_vec_start
										+ gq_itemNum_count * dev_maxDataFeatureID[0]
										+ featureId;

						//if ((groupQuery_boundEntry_vec[featureId_glboalIdx] >= 0)) { //note: if <0, this entry has already been verified
						if (!depressed_isVerified(groupQuery_boundEntry_vec[featureId_glboalIdx])) { //note: check  this entry has already been verified
							//.feature_id = featureId;
							groupQuery_boundEntry_vec[featureId_glboalIdx] = gq_item_BE;
						}

					}
					gq_itemNum_count++;			//exceeds one dimension boundary

				}
				gq_winNum_count++;	//group query with at least one window query
				EC_sw_reverse_id_start = (EC_sw_reverse_id_start - dev_winDimension[0] +dev_maxWindowNumber_perGroupQuery[0]) % dev_maxWindowNumber_perGroupQuery[0];//loop with module of dev_maxWindowNumber_perGroup[0]
			}
		}
	}

}



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
__global__ void compute_groupQuery_lowerBound(
		windowQuery_boundEntry* windowQuery_bound_vec, //input: record the lower bound and upper bound of each window query
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		groupQuery_boundEntry* groupQuery_boundEntry_vec //output: record the lower bound and upper bound of each group query
		) {



	//each block takes one equal class of one item query,  (the number of ECs is equal to window dimension)

	int group_qid = blockIdx.x / (dev_winDimension[0]); //each equal class (EC) is corresponding to multiple qury item in one group query


	//one kernel takes one Equal Classes,the number of ECs per group is equal to window dimension
	//note: we compute the EC by the last sliding windows, i.e. the time series points with large time are within the first windows of ECs.
	int ec_id = blockIdx.x % dev_winDimension[0];
	GroupQuery_info* gquery = d_groupQuery_info_set[group_qid];

	//start of per window query. The number of candidate features for each group queries is number of windows features X window dims
	//Each EC process different segments of all the candidates
	//each group query has multiple different dimensions
	int groupQuery_boundEntry_vec_start = group_qid * dev_maxDataFeatureID[0] * (dev_item_num_perGroupQuery[0]);	//starting position to store result, jumping with window dimension
	int windowQuery_boundEntry_vec_start = group_qid * dev_maxWindowNumber_perGroupQuery[0] * (dev_maxWinFeatureID[0]);// //starting position to access lower bound for each EC

	//load the number of dimension into shared memeory

	extern __shared__ uint gq_sm[];

	uint* gq_item_dimesions = gq_sm;
	//uint* gq_Item_finished = &gq_sm[dev_item_num_perGroupQuery[0]];

	if (threadIdx.x < dev_item_num_perGroupQuery[0]) {
		gq_item_dimesions[threadIdx.x] = gquery->item_dimensions_vec[threadIdx.x];
		//gq_Item_finished[threadIdx.x] = gquery->depressed_item_query_finished[threadIdx.x];

	}
	__syncthreads();

	//uint max_unfinished_ItemId = check_groupQuery_maxUnfinishedItem(
	//		gq_Item_finished);
	//uint max_unfinished_ItemId = dev_item_num_perGroupQuery[0] - 1;

	int gq_maxDimension = gq_item_dimesions[gquery->item_num - 1];

	// |------------------------------------------------------------|
	// |<--------------------------gq_maxDimension----------------->|
	// ||-----| |---------| |----------------| |--------------| |---|
	//                                                           EC starts from right
	//uint ec_index_leftBoundary = (gq_maxDimension - ec_id) / dev_winDimension[0]
	//		- (0 == ((gq_maxDimension - ec_id) % dev_winDimension[0]));

	int round = (*dev_maxWinFeatureID) / blockDim.x
			+ ((*dev_maxWinFeatureID) % blockDim.x != 0);//scan every elements, one element corresponding to multiple window queries within one EC and for different dimensions
	for (int i = 0; i < round; i++) {

		uint ec_index = i * blockDim.x + threadIdx.x;// + ec_index_leftBoundary; //within all time series features,stop to exceed boundary

		if (ec_index < dev_maxWinFeatureID[0] - (ec_id != 0)) //not exceed boundary
		{

			//compute lower bound for feature id for all window queries within one EC
			int gq_itemNum_count = 0;
			int gq_winNum_count = 0;
			groupQuery_boundEntry gq_item_BE = 0;

			//int EC_slidingWindow_Id_start = (gquery->startWindowId + ec_id)
			//		% dev_maxWindowNumber_perGroupQuery[0]; //loop with module of dev_maxWindowNumber_perGroup[0]

			int EC_slidingWindow_id_end = (gquery->startWindowId+dev_maxWindowNumber_perGroupQuery[0]-1)%dev_maxWindowNumber_perGroupQuery[0];//the last sliding window id of this group query
			int EC_sw_reverse_id_start = (dev_maxWindowNumber_perGroupQuery[0] +EC_slidingWindow_id_end - ec_id)%dev_maxWindowNumber_perGroupQuery[0];//the Loop hash, plus mod first to avoid the minus value
					//%dev_maxWindowNumber_perGroupQuery[0];//

			//gq_j is the length of extended query
			for (uint gq_j = ec_id + dev_winDimension[0];//lenght of the extened query inclusive the ec offset part
					gq_j <= gq_maxDimension	&&
					gq_itemNum_count <= (dev_item_num_perGroupQuery[0] - 1)
					&& ec_index>= (gq_winNum_count+ (ec_id!=0));//
					gq_j +=	dev_winDimension[0]) {

				int disWin_index = windowQuery_boundEntry_vec_start
						+ EC_sw_reverse_id_start * dev_maxWinFeatureID[0]
						+ ec_index - gq_winNum_count;

				gq_item_BE += windowQuery_bound_vec[disWin_index];


				if ((gq_j + dev_winDimension[0] > gq_item_dimesions[gq_itemNum_count])
						&& gq_j <= gq_item_dimesions[gq_itemNum_count]) {

					//bool itemQueryFinished = isItemQueryFinished(gq_Item_finished, gq_itemNum_count);
					//if (!itemQueryFinished) {

						int featureId = (ec_index - gq_winNum_count) * dev_winDimension[0]
								- ((dev_winDimension[0] - ec_id) % dev_winDimension[0]);// ec_index>= (gq_winNum_count+ (ec_id!=0) is to avoid exceeding boundary

						int featureId_glboalIdx = groupQuery_boundEntry_vec_start
										+ gq_itemNum_count * dev_maxDataFeatureID[0]
										+ featureId;


						//if (!depressed_isVerified(groupQuery_boundEntry_vec[featureId_glboalIdx])) { //note: check  this entry has already been verified
							//.feature_id = featureId;
							groupQuery_boundEntry_vec[featureId_glboalIdx] = gq_item_BE;
						//}

					//}
					gq_itemNum_count++;			//exceeds one dimension boundary

				}


				gq_winNum_count++;	//group query with at least one window query
				EC_sw_reverse_id_start = (EC_sw_reverse_id_start - dev_winDimension[0] +dev_maxWindowNumber_perGroupQuery[0]) % dev_maxWindowNumber_perGroupQuery[0];//loop with module of dev_maxWindowNumber_perGroup[0]
			}
		}
	}

}



/**
 * edit zhou jingbo
 * TODO :  this function is to compute the lower bound for each possible feature with one query group. One query group compose with multiple slidig windows.
 * One set of disjoint sliding windows within query group forms one  equal class. We use one kernel to take one equal class
 *
 *
 *	enhancedLowerBound_sel:
 *						0: use d2q
 *						1: use q2d
 *						2: use max(d2q,q2d)
 *
 * note: 1. each block takes one equal class of one query
 **		 2.	number of blocks: total number of group queries multiply window dimension (the number of ECs is equal to window dimension
 **		 3. create the sliding windows in reverse order (from tailed to head) for group query.
 *
 */
__global__ void compute_groupQuery_enhancedLowerBound(
		windowQuery_boundEntry* windowQuery_bound_vec_d2q, //input: record the lower bound and upper bound of each window query with d2q
		windowQuery_boundEntry* windowQuery_bound_vec_q2d, //input: record the lower bound and upper bound of each window query with q2d
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		groupQuery_boundEntry* groupQuery_boundEntry_vec, //output: record the lower bound and upper bound of each group query
		int enhancedLowerBound_sel
		) {



	//each block takes one equal class of one item query,  (the number of ECs is equal to window dimension)

	int group_qid = blockIdx.x / (dev_winDimension[0]); //each equal class (EC) is corresponding to multiple qury item in one group query


	//one kernel takes one Equal Classes,the number of ECs per group is equal to window dimension
	//note: we compute the EC by the last sliding windows, i.e. the time series points with large time are within the first windows of ECs.
	int ec_id = blockIdx.x % dev_winDimension[0];
	GroupQuery_info* gquery = d_groupQuery_info_set[group_qid];

	//start of per window query. The number of candidate features for each group queries is number of windows features X window dims
	//Each EC process different segments of all the candidates
	//each group query has multiple different dimensions
	int groupQuery_boundEntry_vec_start = group_qid * dev_maxDataFeatureID[0] * (dev_item_num_perGroupQuery[0]);	//starting position to store result, jumping with window dimension
	int windowQuery_boundEntry_vec_start = group_qid * dev_maxWindowNumber_perGroupQuery[0] * (dev_maxWinFeatureID[0]);// //starting position to access lower bound for each EC

	//load the number of dimension into shared memeory

	extern __shared__ uint gq_sm[];

	uint* gq_item_dimesions = gq_sm;
	//uint* gq_Item_finished = &gq_sm[dev_item_num_perGroupQuery[0]];

	if (threadIdx.x < dev_item_num_perGroupQuery[0]) {
		gq_item_dimesions[threadIdx.x] = gquery->item_dimensions_vec[threadIdx.x];
		//gq_Item_finished[threadIdx.x] = gquery->depressed_item_query_finished[threadIdx.x];

	}
	__syncthreads();

	//uint max_unfinished_ItemId = check_groupQuery_maxUnfinishedItem(
	//		gq_Item_finished);
	//uint max_unfinished_ItemId = dev_item_num_perGroupQuery[0] - 1;

	int gq_maxDimension = gq_item_dimesions[gquery->item_num - 1];

	// |------------------------------------------------------------|
	// |<--------------------------gq_maxDimension----------------->|
	// ||-----| |---------| |----------------| |--------------| |---|
	//                                                           EC starts from right
	//uint ec_index_leftBoundary = (gq_maxDimension - ec_id) / dev_winDimension[0]
	//		- (0 == ((gq_maxDimension - ec_id) % dev_winDimension[0]));

	int round = (*dev_maxWinFeatureID) / blockDim.x
			+ ((*dev_maxWinFeatureID) % blockDim.x != 0);//scan every elements, one element corresponding to multiple window queries within one EC and for different dimensions
	for (int i = 0; i < round; i++) {

		uint ec_index = i * blockDim.x + threadIdx.x;// + ec_index_leftBoundary; //within all time series features,stop to exceed boundary

		if (ec_index < dev_maxWinFeatureID[0] - (ec_id != 0)) //not exceed boundary
		{

			//compute lower bound for feature id for all window queries within one EC
			int gq_itemNum_count = 0;
			int gq_winNum_count = 0;
			groupQuery_boundEntry gq_item_BE_d2q = 0;
			groupQuery_boundEntry gq_item_BE_q2d = 0;

			//int EC_slidingWindow_Id_start = (gquery->startWindowId + ec_id)
			//		% dev_maxWindowNumber_perGroupQuery[0]; //loop with module of dev_maxWindowNumber_perGroup[0]

			int EC_slidingWindow_id_end = (gquery->startWindowId+dev_maxWindowNumber_perGroupQuery[0]-1)%dev_maxWindowNumber_perGroupQuery[0];//the last sliding window id of this group query
			int EC_sw_reverse_id_start = (dev_maxWindowNumber_perGroupQuery[0] +EC_slidingWindow_id_end - ec_id)%dev_maxWindowNumber_perGroupQuery[0];//the Loop hash, plus mod first to avoid the minus value
					//%dev_maxWindowNumber_perGroupQuery[0];//

			//gq_j is the length of extended query
			for (uint gq_j = ec_id + dev_winDimension[0];//lenght of the extened query inclusive the ec offset part
					gq_j <= gq_maxDimension	&&
					gq_itemNum_count <= (dev_item_num_perGroupQuery[0] - 1)
					&& ec_index>= (gq_winNum_count+ (ec_id!=0));//
					gq_j +=	dev_winDimension[0]) {

				int disWin_index = windowQuery_boundEntry_vec_start
						+ EC_sw_reverse_id_start * dev_maxWinFeatureID[0]
						+ ec_index - gq_winNum_count;

				gq_item_BE_d2q += windowQuery_bound_vec_d2q[disWin_index];

				//if(enhancedLowerBound_sel){
				gq_item_BE_q2d+=windowQuery_bound_vec_q2d[disWin_index];
				//}


				if ((gq_j + dev_winDimension[0] > gq_item_dimesions[gq_itemNum_count])
						&& gq_j <= gq_item_dimesions[gq_itemNum_count]) {

					//bool itemQueryFinished = isItemQueryFinished(gq_Item_finished, gq_itemNum_count);
					//if (!itemQueryFinished) {

						int featureId = (ec_index - gq_winNum_count) * dev_winDimension[0]
								- ((dev_winDimension[0] - ec_id) % dev_winDimension[0]);// ec_index>= (gq_winNum_count+ (ec_id!=0) is to avoid exceeding boundary

						int featureId_glboalIdx = groupQuery_boundEntry_vec_start
										+ gq_itemNum_count * dev_maxDataFeatureID[0]
										+ featureId;


						//if (!depressed_isVerified(groupQuery_boundEntry_vec[featureId_glboalIdx])) { //note: check  this entry has already been verified
							//.feature_id = featureId;

						switch(enhancedLowerBound_sel){
						case 0:{
							groupQuery_boundEntry_vec[featureId_glboalIdx] = gq_item_BE_d2q;
							break;
						}
						case 1:{
							groupQuery_boundEntry_vec[featureId_glboalIdx] = gq_item_BE_q2d;
							break;
						}
						case 2:{
							groupQuery_boundEntry_vec[featureId_glboalIdx] = fmaxf(gq_item_BE_d2q,gq_item_BE_q2d);
							break;
						}
						default:{
							groupQuery_boundEntry_vec[featureId_glboalIdx] = fmaxf(gq_item_BE_d2q,gq_item_BE_q2d);
							break;
						}

						}
//						if(!enhancedLowerBound_sel){
//							groupQuery_boundEntry_vec[featureId_glboalIdx] = gq_item_BE_d2q;
//						}else{
//							groupQuery_boundEntry_vec[featureId_glboalIdx] = fmaxf(gq_item_BE_d2q,gq_item_BE_q2d);
//						}
						//}

					//}
					gq_itemNum_count++;			//exceeds one dimension boundary

				}


				gq_winNum_count++;	//group query with at least one window query
				EC_sw_reverse_id_start = (EC_sw_reverse_id_start - dev_winDimension[0] +dev_maxWindowNumber_perGroupQuery[0]) % dev_maxWindowNumber_perGroupQuery[0];//loop with module of dev_maxWindowNumber_perGroup[0]
			}
		}
	}

}



/**
 * TODO:
 *  We need to scan the groupQuery_result_vec and check the query status.
 *   First we output the candidate whose lower bound is larger than the threshold of the query items within this group query
 *  In this funciton, we first do prefix count in order to allocate the space to store the candidates. Second, if there is
 *  no output for this query item, we update the query status in "d_groupQuery_info_set"
 *
 *
 *  note:1. one block take one query (one group query may have multiple query items with different dimensnions),
 *  	 	 i.e. block number is equal to groupQuery number multiply the number of query with different dimensions within one group query
 *
 */
__global__ void depressed_prefixCount_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_result_vec, //input: record the lower bound  of each group query
		float* threshold, //input: threshold for each group query items
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_count_vec, //output: compute candidates per thread
		int* blk_count_vec, //output: compute candidates per block (kernel)
		GroupQuery_info** d_groupQuery_info_set //output:if the candidates are empty, set true for this group query (item).
		) {


		int gid = blockIdx.x / dev_item_num_perGroupQuery[0];
		int gItemId = blockIdx.x % dev_item_num_perGroupQuery[0];
		GroupQuery_info* gq_info = d_groupQuery_info_set[gid];

		if(gq_info->depressed_isItemQueryFinished(gItemId)){
			return;//if this group query item finished, return
		}


		extern __shared__ int sm_countVec[]; //shared memory dynamic allocate with the size of blockDim.x, i.e. number of threads

		int gq_item_dimension = gq_info->getItemDimension(gItemId);
		int blade_id = gq_info->blade_id;
		uint ts_blade_start = (blade_id == 0) ? 0:ts_data_blade_endIdx[blade_id-1];
		uint ts_blade_len = ts_data_blade_endIdx[blade_id] - ts_blade_start- gq_item_dimension+1;

		//int candidate_Len =dev_maxDataFeatureID[0];// dev_maxWinFeatureID[0] * dev_winDimension[0];//total number of features is equal to number of window features multiply number of dimensions.

		int round = (ts_blade_len) / blockDim.x
				+ ((ts_blade_len) % blockDim.x != 0); //scan every elements,

		int group_query_start = blockIdx.x * dev_maxDataFeatureID[0];//note : we must have  ts_blade_len <dev_maxDataFeatureID[0]
		float gq_item_threshold = threshold[blockIdx.x];

		int t_count = 0;
		int t_count_idx = blockIdx.x * blockDim.x + threadIdx.x;

		for (int i = 0; i < round; i++) {
			int index = i * blockDim.x + threadIdx.x;
			if (index < ts_blade_len) {
				groupQuery_boundEntry gbe =
						groupQuery_result_vec[index + group_query_start];

				if (gbe < gq_item_threshold&&(!depressed_isVerified(gbe))){//(gbe >=0)) {//if cte<=-1, this candidate has already been verified
					t_count++;

				}
			}
		}

		thread_count_vec[t_count_idx] = t_count;
		sm_countVec[threadIdx.x] =  t_count;
		__syncthreads();

		//note: the size of sm_countVec is just the number of threads per block
		uint pre_s;
		for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
			pre_s = blockDim.x;
			if (threadIdx.x < s) {
				sm_countVec[threadIdx.x] += sm_countVec[threadIdx.x + s];
			}

			if(pre_s%2 == 1 && threadIdx.x == 0){
				sm_countVec[0] +=sm_countVec[2*s];
			}
			pre_s=s;
		}
		__syncthreads();

		if (threadIdx.x == 0) {

			blk_count_vec[blockIdx.x] = sm_countVec[0];
			if(0 == sm_countVec[0]){
				gq_info->depressed_setItemQueryFinished(gItemId);
			}
		}

}




/**
 * TODO:
 *  We need to scan the groupQuery_result_vec and check the query status.
 *   First we output the candidate whose lower bound is larger than the threshold of the query items within this group query
 *  In this funciton, we first do prefix count in order to allocate the space to store the candidates. Second, if there is
 *  no output for this query item, we update the query status in "d_groupQuery_info_set"
 *
 *
 *  note:1. one block take one query (one group query may have multiple query items with different dimensnions),
 *  	 	 i.e. block number is equal to groupQuery number multiply the number of query with different dimensions within one group query
 *
 */
__global__ void prefixCount_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_result_vec, //input: record the lower bound  of each group query
		float* threshold, //input: threshold for each group query items
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_count_vec, //output: compute candidates per thread
		int* blk_count_vec, //output: compute candidates per block (kernel)
		GroupQuery_info** d_groupQuery_info_set //output:if the candidates are empty, set true for this group query (item).
		) {


		int gid = blockIdx.x / dev_item_num_perGroupQuery[0];
		int gItemId = blockIdx.x % dev_item_num_perGroupQuery[0];
		GroupQuery_info* gq_info = d_groupQuery_info_set[gid];

		extern __shared__ int sm_countVec[]; //shared memory dynamic allocate with the size of blockDim.x, i.e. number of threads

		int gq_item_dimension = gq_info->getItemDimension(gItemId);
		int blade_id = gq_info->blade_id;
		uint ts_blade_start = (blade_id == 0) ? 0:ts_data_blade_endIdx[blade_id-1];
		uint ts_blade_len = ts_data_blade_endIdx[blade_id] - ts_blade_start- gq_item_dimension+1;

		//int candidate_Len =dev_maxDataFeatureID[0];// dev_maxWinFeatureID[0] * dev_winDimension[0];//total number of features is equal to number of window features multiply number of dimensions.

		int round = (ts_blade_len) / blockDim.x
				+ ((ts_blade_len) % blockDim.x != 0); //scan every elements,

		int group_query_start = blockIdx.x * dev_maxDataFeatureID[0];//note : we must have  ts_blade_len <dev_maxDataFeatureID[0]
		float gq_item_threshold = threshold[blockIdx.x];

		int t_count = 0;
		int t_count_idx = blockIdx.x * blockDim.x + threadIdx.x;

		for (int i = 0; i < round; i++) {
			int index = i * blockDim.x + threadIdx.x;
			if (index < ts_blade_len) {
				groupQuery_boundEntry gbe =
						groupQuery_result_vec[index + group_query_start];


				//if (gbe < gq_item_threshold&&(!depressed_isVerified(gbe))){//(gbe >=0)) {//if cte<=-1, this candidate has already been verified
				if (gbe <= gq_item_threshold){//(gbe >=0)) {//if cte<=-1, this candidate has already been verified


					t_count++;
				}
			}
		}

		thread_count_vec[t_count_idx] = t_count;

		sm_countVec[threadIdx.x] =  t_count;
		__syncthreads();


		//note: the size of sm_countVec is just the number of threads per block, and the number is times of 2
		//uint pre_s=blockDim.x;
		for (uint s = blockDim.x / 2; s > 0; s >>= 1) {

			if (threadIdx.x < s) {
				sm_countVec[threadIdx.x] += sm_countVec[threadIdx.x + s];
			}

			//if(pre_s%2 == 1 && threadIdx.x == 0){
			//	sm_countVec[0] +=sm_countVec[2*s];
			//}
			//pre_s=s;
			__syncthreads();
		}

	__syncthreads();

	if (threadIdx.x == 0) {

		blk_count_vec[blockIdx.x] = sm_countVec[0];
	}


}



/**edit zhou jingbo
 * TODO :  this function is to scan the groupQuery_result_vec and output the candidate whose lower bound is larger than the threshold of this group query
 *
 * note:1. one block take one query items, i.e. block number is equal to the  total query items of all groupQuery
 *
 */
__global__ void depressed_output_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_item_boundEntry_vec, //input: record the lower bound of each group query
		float* threshold,//input: threshold for each group query
		GroupQuery_info** d_groupQuery_info_set, //input:check wheter this group query (item) is finished
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_result_vec_endIdx, //output: compute candidates per thread
		CandidateEntry* candidate_result//output: store the candidates for all group queries
	){

		int gid = blockIdx.x / dev_item_num_perGroupQuery[0];
		int gItemId = blockIdx.x % dev_item_num_perGroupQuery[0];
		GroupQuery_info* gq_info = d_groupQuery_info_set[gid];
		if(gq_info->depressed_isItemQueryFinished(gItemId)){
				return;//if this group query item finished, return
		}

		uint pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint start_idx = (pos == 0) ? 0 : thread_result_vec_endIdx[pos-1];
		uint end_idx = thread_result_vec_endIdx[pos];
		if(end_idx==start_idx) return;//there is not work for this thread. It has been finished

		int gq_item_dimension = gq_info->getItemDimension(gItemId);
		int blade_id = gq_info->blade_id;
		uint ts_blade_start = (blade_id == 0) ? 0:ts_data_blade_endIdx[blade_id-1];
		uint ts_blade_len = ts_data_blade_endIdx[blade_id] - ts_blade_start- gq_item_dimension+1;


		//int candidate_Len =  dev_maxWinFeatureID[0]*dev_winDimension[0];
		int group_query_start = blockIdx.x*dev_maxDataFeatureID[0];//note : we must have  ts_blade_len <dev_maxDataFeatureID[0]
		int round = (ts_blade_len) / blockDim.x
						+ ((ts_blade_len) % blockDim.x != 0);//scan every elements,
		float gquery_threshold = threshold[blockIdx.x];


		for(int i=0;i<round;i++){
			uint index = i* blockDim.x + threadIdx.x;
			if(index<ts_blade_len){

				groupQuery_boundEntry cte = groupQuery_item_boundEntry_vec[index+group_query_start];
				//if(cte < gquery_threshold && (cte>=0)){
				if(cte < gquery_threshold && !depressed_isVerified(cte)){
					CandidateEntry ce (index,cte);
					candidate_result[start_idx] = ce;
					start_idx++;
				}
			}
		}
}



/**edit zhou jingbo
 * TODO :  this function is to scan the groupQuery_result_vec and output the candidate whose lower bound is larger than the threshold of this group query
 *
 * note:1. one block take one query items, i.e. block number is equal to the  total query items of all groupQuery
 *
 */
__global__ void output_groupQueryCandidates_Threshold(
		groupQuery_boundEntry* groupQuery_item_boundEntry_vec, //input: record the lower bound of each group query
		float* threshold,//input: threshold for each group query
		GroupQuery_info** d_groupQuery_info_set, //input:check wheter this group query (item) is finished
		const int* ts_data_blade_endIdx, //input: get the bound for scan of each query item,  the endidx for each blades (i.e. the boundary of different blades),
		int* thread_result_vec_endIdx, //output: compute candidates per thread
		CandidateEntry* candidate_result//output: store the candidates for all group queries
	){

		int gid = blockIdx.x / dev_item_num_perGroupQuery[0];
		int gItemId = blockIdx.x % dev_item_num_perGroupQuery[0];
		GroupQuery_info* gq_info = d_groupQuery_info_set[gid];

		uint pos = blockIdx.x * blockDim.x + threadIdx.x;
		uint start_idx = (pos == 0) ? 0 : thread_result_vec_endIdx[pos-1];
		uint end_idx = thread_result_vec_endIdx[pos];
		if(end_idx==start_idx) return;//there is not work for this thread. It has been finished

		int gq_item_dimension = gq_info->getItemDimension(gItemId);
		int blade_id = gq_info->blade_id;
		uint ts_blade_start = (blade_id == 0) ? 0:ts_data_blade_endIdx[blade_id-1];
		uint ts_blade_len = ts_data_blade_endIdx[blade_id] - ts_blade_start- gq_item_dimension+1;


		//int candidate_Len =  dev_maxWinFeatureID[0]*dev_winDimension[0];
		int group_query_start = blockIdx.x*dev_maxDataFeatureID[0];//note : we must have  ts_blade_len <dev_maxDataFeatureID[0]
		int round = (ts_blade_len) / blockDim.x
						+ ((ts_blade_len) % blockDim.x != 0);//scan every elements,
		float gquery_threshold = threshold[blockIdx.x];


		for(int i=0;i<round;i++){
			uint index = i* blockDim.x + threadIdx.x;
			if(index<ts_blade_len){

				groupQuery_boundEntry cte = groupQuery_item_boundEntry_vec[index+group_query_start];
				//if(cte < gquery_threshold && (cte>=0)){
				//if(cte < gquery_threshold && !depressed_isVerified(cte)){
				if(cte <= gquery_threshold){

					CandidateEntry ce(index,cte);
					candidate_result[start_idx] = ce;
					//candidate_result[start_idx].feature_id=index;
					//candidate_result[start_idx].dist=cte;

				    start_idx++;

				}
			}
		}
}





/*
 * TODO:
 * 		verify the candidates and update the status of each bound entry (set as VERIFIED_LABEL_VALUE if verified)
 *
 * 		note:
 * 		1. there are two cases for verifying the candidates, controlled by "candidates_number_fixed"
 * 			1): the number of candidates is fixed number, but sometimes the number of candidates may be less than the fixed number,the actually
 * 			    numer of candidates is stored in "d_groupQuerySet_candidates_idxInfo", the default candidate number is stored in "candidats_number_fixed"
 *
 * 			2): the number of candidates is flexible, the starting position of the candidates for each query item is stored in
 * 			    "d_groupQuerySet_candidates_idxInfo". In this case, candidats_number_fixed is set as 0
 *
 *     2.  the number of blocks is the total queries, i.e. the number of group query multiply the number of item query with different dimensions per group
 *
 */
__global__ void depressed_scan_verifyCandidates(
		const int sc_band, //input: Sakoe-Chiba Band
		const int candidates_number_fixed, //refer to note 1, if candidats_number_fixed == 0, the candidates number is flexible, else, this defines the fixed (maximum) possible number of candidates in d_groupQuerySet_candidates"
		GroupQuery_info** groupQuery_info_set, //input:group query
		const float* d_ts_data, //the time series blade, note: there may be multiple blades
		const int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
		CandidateEntry* groupQuerySet_candidates, //input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
		const int* groupQuerySet_candidates_idxInfo, //input:record the idx for each group query in d_groupQuerySet_Candidates
		groupQuery_boundEntry* groupQuery_boundEntry_vec //output: update the boundEntry after verification and set the verified as true
		) {


	int gq_id = blockIdx.x / dev_item_num_perGroupQuery[0];
	int gq_item_id = blockIdx.x % dev_item_num_perGroupQuery[0];

	GroupQuery_info* gq = groupQuery_info_set[gq_id];
	if (gq->depressed_isItemQueryFinished(gq_item_id))
		return; //if this group query has been finished, return directly

	int gq_item_dimension = gq->item_dimensions_vec[gq_item_id];
	int gq_max_dimension =
			gq->item_dimensions_vec[dev_item_num_perGroupQuery[0] - 1]; //all group queris have the same number of query items (with different dimension)

	//
	//note: create shared memory with the maximum dimension of item query in a group and plus one
	//(dev_groupQuery_maxdimension+1)*sizeof(float),
	extern __shared__ float share_gq_item_data[];
	share_gq_item_data[0] = (float) INT_MAX;

	uint groupQueryItem_feature_Len = dev_maxWinFeatureID[0] * dev_winDimension[0];
	uint groupQueryItem_start = blockIdx.x * groupQueryItem_feature_Len;

	//load query item info into shared memory
	int round_loadQuery = (gq_item_dimension) / blockDim.x + ((gq_item_dimension) % blockDim.x != 0); //scan every elements,



	//data are stored in resevered order, the query with small dimensions align to right side
	int gq_item_dimension_start = gq_max_dimension - gq_item_dimension ;
	for (int i = 0; i < round_loadQuery; i++) {
		int index = i * blockDim.x + threadIdx.x;
		if (index < gq_item_dimension) {
			//note: load data right side first, from tail to head
			share_gq_item_data[index + 1] =	//start from 1, the first element is padded with INT_MAX
					gq->data[gq_item_dimension_start + index]; //note: gq_item_dimStart+index, load the right most data from GroupQuery_info.data
		}
	}


	__syncthreads();


	//compute the DTW for each Candidates
	uint candidates_start,candidates_size;

	if(0 == candidates_number_fixed){// the candidates_number is flexible, load the start and end position

		candidates_start =
			(blockIdx.x == 0) ?
					0 : (groupQuerySet_candidates_idxInfo[blockIdx.x - 1]);
		candidates_size = groupQuerySet_candidates_idxInfo[blockIdx.x] - candidates_start;

	}else{//the candidate number is fixed, load the actual number of candidates, which may be smaller than candidates_number_fixed

		candidates_start = blockIdx.x * candidates_number_fixed;
		candidates_size =  groupQuerySet_candidates_idxInfo[blockIdx.x];

	}

	if(threadIdx.x<candidates_size){

	//load blade id for this group query
	//uint ts_data_start = d_ts_data_blade_endIdx[gq->blade_id];
	uint blade_id = gq->blade_id;
	uint ts_data_start = (0==blade_id)?
							0:(d_ts_data_blade_endIdx[blade_id-1]);
	//uint ts_data_end = d_ts_data_blade_endIdx[blade_id];
	//uint ts_data_len = d_ts_data_blade_endIdx[blade_id] - ts_data_start;


	int round_compDist = (candidates_size) / blockDim.x
			+ ((candidates_size) % blockDim.x != 0); //scan every elements,


	float* candidate_item = new float[gq_item_dimension + 1];
	candidate_item[0] = (float) INT_MAX; //note:first element padded with MAX

	for (int i = 0; i < round_compDist; i++) {
		int index = i * blockDim.x + threadIdx.x;
		if (index < candidates_size) {
			CandidateEntry ce = groupQuerySet_candidates[candidates_start + index];

			for (int j = 0; j < gq_item_dimension; j++) {

				candidate_item[j + 1] = //first element padded with max
						d_ts_data[ts_data_start + ce.feature_id + j]; //load following items

			}

			float dtw_dist = dtw_DP_SCBand(share_gq_item_data, gq_item_dimension, candidate_item, gq_item_dimension, sc_band);

			ce.dist = dtw_dist;

			groupQuerySet_candidates[candidates_start + index] = ce;//write back
			groupQuery_boundEntry_vec[groupQueryItem_start + ce.feature_id] = depressed_VERIFIED_LABEL_VALUE;//-1;//this boundEntry has already been verified
		}
	}

	delete[] candidate_item;
	//}
	}
	return;
}





/*
 * TODO:
 * 		verify the candidates and update the status of each bound entry
 *
 * 		note:
 * 		1. there are two cases for verifying the candidates, controlled by "candidates_number_fixed"
 * 			1): the number of candidates is fixed number, but sometimes the number of candidates may be less than the fixed number,the actually
 * 			    numer of candidates is stored in "d_groupQuerySet_candidates_idxInfo", the default candidate number is stored in "candidats_number_fixed"
 *
 * 			2): the number of candidates is flexible, the starting position of the candidates for each query item is stored in
 * 			    "d_groupQuerySet_candidates_idxInfo". In this case, candidats_number_fixed is set as 0
 *
 *     2.  the number of blocks is the total queries, i.e. the number of group query multiply the number of item query with different dimensions per group
 *
 *     3.  (DONOT set  VERIFIED_LABEL_VALUE)
 *
 */
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
		DIST distFunc) {


	int gq_id = blockIdx.x / dev_item_num_perGroupQuery[0];
	int gq_item_id = blockIdx.x % dev_item_num_perGroupQuery[0];

	GroupQuery_info* gq = groupQuery_info_set[gq_id];
	//if (gq->depressed_isItemQueryFinished(gq_item_id))
	//	return; //if this group query has been finished, return directly


	int gq_item_dimension = gq->item_dimensions_vec[gq_item_id];
	int gq_max_dimension =
			gq->item_dimensions_vec[dev_item_num_perGroupQuery[0] - 1]; //all group queris have the same number of query items (with different dimension)


	//
	//note: create shared memory with the maximum dimension of item query in a group and plus one
	//(dev_groupQuery_maxdimension+1)*sizeof(float),
	extern __shared__ float share_gq_item_data[];
	//share_gq_item_data[0] = (float) INT_MAX;

	//uint groupQueryItem_feature_Len = dev_maxWinFeatureID[0] * dev_winDimension[0];
	//uint groupQueryItem_start = blockIdx.x * groupQueryItem_feature_Len;

	//load query item info into shared memory
	int round_loadQuery = (gq_item_dimension) / blockDim.x + ((gq_item_dimension) % blockDim.x != 0); //scan every elements,


	//data are stored in resevered order, the query with small dimensions align to right side
	int gq_item_dimension_start = gq_max_dimension - gq_item_dimension ;
	for (int i = 0; i < round_loadQuery; i++) {
		int index = i * blockDim.x + threadIdx.x;
		if (index < gq_item_dimension) {
			//note: load data right side first, from tail to head
			share_gq_item_data[index] =
					gq->data[gq_item_dimension_start + index]; //note: gq_item_dimStart+index, load the right most data from GroupQuery_info.data
		}
	}


	__syncthreads();


	//compute the DTW for each Candidates
	int candidates_start,candidates_size;

	if(0 == candidates_number_fixed){// the candidates_number is flexible, load the start and end position

		candidates_start =
			(blockIdx.x == 0) ?
					0 : (groupQuerySet_candidates_idxInfo[blockIdx.x - 1]);
		candidates_size = groupQuerySet_candidates_idxInfo[blockIdx.x] - candidates_start;

	}else{//the candidate number is fixed, load the actual number of candidates, which may be smaller than candidates_number_fixed

		candidates_start = blockIdx.x * candidates_number_fixed;
		candidates_size =  groupQuerySet_candidates_idxInfo[blockIdx.x];

	}


	if(threadIdx.x<candidates_size){

	//load blade id for this group query
	int blade_id = gq->blade_id;
	int ts_data_start = (0==blade_id)?
							0:(d_ts_data_blade_endIdx[blade_id-1]);

	int round_compDist = (candidates_size) / blockDim.x
			+ ((candidates_size) % blockDim.x != 0); //scan every elements,


	for (int i = 0; i < round_compDist; i++) {
		int index = i * blockDim.x + threadIdx.x;
		if (index < candidates_size) {


			int ce_feature_id = groupQuerySet_candidates[candidates_start + index].feature_id;
			float dist = distFunc.dist(share_gq_item_data, 0, d_ts_data, ts_data_start + ce_feature_id, gq_item_dimension);

			groupQuerySet_candidates[candidates_start + index].dist =dist;

		}
	}

}

}


template __global__ void scan_verifyCandidates<Dtw_SCBand_Func_modulus_flt>(
		const int sc_band, //input: Sakoe-Chiba Band
		const int candidates_number_fixed, //refer to note 1, if candidats_number_fixed == 0, the candidates number is flexible, else, this defines the fixed (maximum) possible number of candidates in d_groupQuerySet_candidates"
		GroupQuery_info** groupQuery_info_set, //input:group query
		const float* d_ts_data, //the time series blade, note: there may be multiple blades
		const int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
		CandidateEntry* groupQuerySet_candidates, //input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
		const int* groupQuerySet_candidates_idxInfo, //input:record the idx for each group query in d_groupQuerySet_Candidates
		//groupQuery_boundEntry* groupQuery_boundEntry_vec, //output: update the boundEntry after verification and set the verified as true
		Dtw_SCBand_Func_modulus_flt distFunc);




/*
 * TODO:
 * 		verify the candidates and update the status of each bound entry
 *
 * 		note:
 * 		1. there are two cases for verifying the candidates, controlled by "candidates_number_fixed"
 * 			1): the number of candidates is fixed number, but sometimes the number of candidates may be less than the fixed number,the actually
 * 			    numer of candidates is stored in "d_groupQuerySet_candidates_idxInfo", the default candidate number is stored in "candidats_number_fixed"
 *
 * 			2): the number of candidates is flexible, the starting position of the candidates for each query item is stored in
 * 			    "d_groupQuerySet_candidates_idxInfo". In this case, candidats_number_fixed is set as 0
 *
 *     2.  the number of blocks is the total queries, i.e. the number of group query multiply the number of item query with different dimensions per group
 *
 *     3.  (DONOT set  VERIFIED_LABEL_VALUE)
 *
 */
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
		DIST distFunc) {


	int gq_id = (blockIdx.x/(threadPerBlock/mergeThreadInterval))/ dev_item_num_perGroupQuery[0];
	int gq_item_id = (blockIdx.x/(threadPerBlock/mergeThreadInterval)) % dev_item_num_perGroupQuery[0];

	GroupQuery_info* gq = groupQuery_info_set[gq_id];
	//if (gq->depressed_isItemQueryFinished(gq_item_id))
	//	return; //if this group query has been finished, return directly


	int gq_item_dimension = gq->item_dimensions_vec[gq_item_id];
	int gq_max_dimension =
			gq->item_dimensions_vec[dev_item_num_perGroupQuery[0] - 1]; //all group queris have the same number of query items (with different dimension)


	//
	//note: create shared memory with the maximum dimension of item query in a group and plus one
	//(dev_groupQuery_maxdimension+1)*sizeof(float),
	extern __shared__ float share_gq_item_data[];
	//share_gq_item_data[0] = (float) INT_MAX;

	//uint groupQueryItem_feature_Len = dev_maxWinFeatureID[0] * dev_winDimension[0];
	//uint groupQueryItem_start = blockIdx.x * groupQueryItem_feature_Len;

	//load query item info into shared memory
	int round_loadQuery = (gq_item_dimension) / blockDim.x + ((gq_item_dimension) % blockDim.x != 0); //scan every elements,


	//data are stored in reserved order, the query with small dimensions align to right side
	int gq_item_dimension_start = gq_max_dimension - gq_item_dimension ;
	for (int i = 0; i < round_loadQuery; i++) {
		int index = i * blockDim.x + threadIdx.x;
		if (index < gq_item_dimension) {
			//note: load data
			share_gq_item_data[index] =
					gq->data[gq_item_dimension_start + index]; //note: gq_item_dimStart+index, load the right most data from GroupQuery_info.data
		}
	}


	__syncthreads();


	//compute the DTW for each Candidates
	int candidates_start,candidates_size;

		candidates_start =(blockIdx.x == 0) ?
					0 : (groupQuerySet_candiates_thread_endIdx[(blockIdx.x)*mergeThreadInterval-1]);
		candidates_size = groupQuerySet_candiates_thread_endIdx[(blockIdx.x+1)*mergeThreadInterval-1] - candidates_start;


	if(threadIdx.x<candidates_size){

	//load blade id for this group query
	int blade_id = gq->blade_id;
	int ts_data_start = (0==blade_id)?
							0:(d_ts_data_blade_endIdx[blade_id-1]);

	int round_compDist = (candidates_size) / blockDim.x
			+ ((candidates_size) % blockDim.x != 0); //scan every elements,


	for (int i = 0; i < round_compDist; i++) {
		int index = i * blockDim.x + threadIdx.x;
		if (index < candidates_size) {

			int ce_feature_id = groupQuerySet_candidates[candidates_start + index].feature_id;
			float dist = distFunc.dist(share_gq_item_data, 0, d_ts_data, ts_data_start + ce_feature_id, gq_item_dimension);

			groupQuerySet_candidates[candidates_start + index].dist =dist;

		}
	}

}

}

template __global__ void scan_verifyCandidates_perThreadGenerated<Dtw_SCBand_Func_modulus_flt>(
		const int sc_band, //input: Sakoe-Chiba Band
		GroupQuery_info** groupQuery_info_set, //input:group query
		const float* d_ts_data, //the time series blade, note: there may be multiple blades
		const int* d_ts_data_blade_endIdx, //the endidx for each blades (i.e. the boundary of different blades)
		CandidateEntry* groupQuerySet_candidates, //input and output: retrieve time series with the d_groupQuerySet_Candidates_id and compute the dist into d_groupQuerySet_Candidates.dist
		const int* groupQuerySet_candiates_thread_endIdx,
		const int threadPerBlock, const int mergeThreadInterval,
		//groupQuery_boundEntry* groupQuery_boundEntry_vec, //output: update the boundEntry after verification and set the verified as true
		Dtw_SCBand_Func_modulus_flt distFunc);



/**
 * TODO:
 * sort the data from ascending order, from small to large
 */
__device__ void inline blk_sort_candidateEntry(CandidateEntry* candidates_array, CandidateEntry* temp_array, int size)
{

	for (uint base = 1; base < size; base <<= 1) {
		int sort_round = size / blockDim.x + (size % blockDim.x != 0);
		for (int ri = 0; ri < sort_round; ri++) {
			int idx = ri * blockDim.x + threadIdx.x;
			uint index = 2 * base * idx;

			if (index < size) {
				int start_idx_x = index; //threadIdx.x;
				int end_idx_x = start_idx_x + base;
				end_idx_x = end_idx_x < size ? end_idx_x : size;

				int start_idx_y = end_idx_x;
				int end_idx_y = start_idx_y + base;
				end_idx_y = end_idx_y < size ? end_idx_y : size;

				int x_pointer = start_idx_x;
				int y_pointer = start_idx_y;

				int output_ptr = x_pointer;
				while (x_pointer < end_idx_x || y_pointer < end_idx_y) {
					if (x_pointer >= end_idx_x)
						temp_array[output_ptr++] = candidates_array[y_pointer++];
					else if (y_pointer >= end_idx_y)
						temp_array[output_ptr++] = candidates_array[x_pointer++];
					else if (candidates_array[x_pointer].dist > candidates_array[y_pointer].dist)//key operation
						temp_array[output_ptr++] = candidates_array[y_pointer++];
					else
						temp_array[output_ptr++] = candidates_array[x_pointer++];
				}

				while (x_pointer < end_idx_x) {
					temp_array[output_ptr++] = candidates_array[x_pointer++];
				}
				while (y_pointer < end_idx_y) {
					temp_array[output_ptr++] = candidates_array[y_pointer++];
				}
			}
		}
		__syncthreads();

		//copy back to shared_array
		int round = size / blockDim.x + (size % blockDim.x != 0);
		for (int i = 0; i < round; i++) {
			int idx = i * blockDim.x + threadIdx.x;
			if (idx < size) {
				candidates_array[idx] = temp_array[idx];
			}
		}
	}
	__syncthreads();
}

//auxiliary function for merge k-selection
__device__ void inline select_candidateEntry_mergeSort_aux_sortSegment(CandidateEntry* candidates_array, CandidateEntry* temp_array, int size, int& _sg_len, bool sort_asc)
{

	uint  base = 1;
	for(; base < size && base<_sg_len; base<<=1){

		int sort_round = size/blockDim.x + (size % blockDim.x  != 0);
		for(int ri=0;ri<sort_round;ri++){
		int idx = ri * blockDim.x  + threadIdx.x;
		uint index = 2*base*idx;
		//if(threadIdx.x % (2*base) == 0){
		if(index<size){
			int start_idx_x = index;//threadIdx.x;
			int end_idx_x = start_idx_x + base;
			end_idx_x = end_idx_x < size ? end_idx_x : size;

			int start_idx_y = end_idx_x;
			int end_idx_y = start_idx_y + base;
			end_idx_y = end_idx_y < size ? end_idx_y : size;

			int x_pointer = start_idx_x;
			int y_pointer = start_idx_y;

			int output_ptr = x_pointer;
			while(x_pointer < end_idx_x || y_pointer < end_idx_y)
			{
				if (x_pointer >= end_idx_x)
					temp_array[output_ptr++] = candidates_array[y_pointer++];
				else if (y_pointer >= end_idx_y)
					temp_array[output_ptr++] = candidates_array[x_pointer++];
				else if(sort_asc) {//select the smallest one, sorting the lower bound with ascending order
				if (candidates_array[x_pointer].dist > candidates_array[y_pointer].dist)
					temp_array[output_ptr++] = candidates_array[y_pointer++];
				else
					temp_array[output_ptr++] = candidates_array[x_pointer++];
				}else{
					if (candidates_array[x_pointer].dist < candidates_array[y_pointer].dist)
						temp_array[output_ptr++] = candidates_array[y_pointer++];
					else
						temp_array[output_ptr++] = candidates_array[x_pointer++];
				}
			}
		}
		}
		__syncthreads();


		int round = size/blockDim.x + (size % blockDim.x  != 0);
		for(int i = 0; i < round; i++){
			int idx = i * blockDim.x  + threadIdx.x;
			if(idx<size){
			candidates_array[idx] = temp_array[idx];
			}
		}

		//__syncthreads();
		//base *= 2;
	}
	_sg_len =base;
}

//copy from A to B
__device__ void inline select_candidateEnry_mergeSort_aux_cp(CandidateEntry* A, CandidateEntry *B, int size){

	int round = size/blockDim.x + (size % blockDim.x  != 0);
		for(int i = 0; i < round; i++){
			int idx = i * blockDim.x  + threadIdx.x;
			if(idx<size){
			B[idx] = A[idx];
			}
		}

}

__device__ void inline select_candidateEnry_mergeSort_aux_compact(CandidateEntry* A, CandidateEntry* _B, int& compact_size, const int size, const int k, int base){
	__syncthreads();
	compact_size = (size / base) * k ;

	if(size%base<=k){
		compact_size+=size%base;

	}else{
		compact_size+=k;
	}


	int round = size / blockDim.x + (size % blockDim.x != 0);

	for (int i = 0; i < round; i++) {

		int idx = i * blockDim.x + threadIdx.x;

		if(idx<size){

		int si = 0;
		if (idx % base < k) {

			si = (idx / base) * k + idx % base;

		} else {

			si = idx - (idx / base+1) * k + compact_size;

		}

		_B[si] = A[idx];

		}

	}
	__syncthreads();
}


__device__ void inline select_candidateEntry_mergeSort(
		CandidateEntry* candidates_array,//input and output: the selected k elements are put in the head part
		CandidateEntry* temp_array,
		const int size,
		const int k,
		bool sort_asc)
{

	//create share memeory


	int sg_len = k;
	int compact_size=0;

	select_candidateEntry_mergeSort_aux_sortSegment(candidates_array, temp_array, size, sg_len,sort_asc);
	select_candidateEnry_mergeSort_aux_compact(candidates_array, temp_array, compact_size, size, k, sg_len);	//compact to temp_array
	select_candidateEnry_mergeSort_aux_cp(temp_array, candidates_array, size);

	int num = size / sg_len + (size % sg_len != 0);

	for (; num > 1; num =(num/2+num%2)) {

		int select_round = compact_size/blockDim.x + (compact_size % blockDim.x  != 0);

		for(int sri=0;sri<select_round;sri++){
			uint idx = sri * blockDim.x  + threadIdx.x;
			uint index = idx * 2 * k;

		if (index < compact_size) {

			int start_idx_x = index;		//threadIdx.x;
			int end_idx_x = start_idx_x + k;
			end_idx_x = end_idx_x < compact_size ? end_idx_x : compact_size;

			int start_idx_y = end_idx_x;
			int end_idx_y = start_idx_y + k;
			end_idx_y = end_idx_y < compact_size ? end_idx_y : compact_size;

			int x_pointer = start_idx_x;
			int y_pointer = start_idx_y;

			int output_ptr = x_pointer;
			while ((x_pointer < end_idx_x || y_pointer < end_idx_y)
					&& (output_ptr < start_idx_x + k)) {
				if (x_pointer >= end_idx_x)
					temp_array[output_ptr++] = candidates_array[y_pointer++];
				else if (y_pointer >= end_idx_y)
					temp_array[output_ptr++] = candidates_array[x_pointer++];
				else if(sort_asc){////select the smallest one, sorting the lower bound with ascending order
					if (candidates_array[x_pointer].dist > candidates_array[y_pointer].dist)
					temp_array[output_ptr++] = candidates_array[y_pointer++];
					else
					temp_array[output_ptr++] = candidates_array[x_pointer++];
				} else{//select the smallest one, sorting the upper bound with ascending order
					if (candidates_array[x_pointer].dist > candidates_array[y_pointer].dist)
						temp_array[output_ptr++] = candidates_array[y_pointer++];
					else
						temp_array[output_ptr++] = candidates_array[x_pointer++];
				}
			}

			while (x_pointer < end_idx_x) {
				temp_array[output_ptr++] = candidates_array[x_pointer++];
			}
			while (y_pointer < end_idx_y) {
				temp_array[output_ptr++] = candidates_array[y_pointer++];
			}
		}
		}

			__syncthreads();

			int k_b_len = compact_size;
			select_candidateEnry_mergeSort_aux_compact(temp_array, candidates_array, compact_size, k_b_len, k, 2*k);	//compact to shared_array

	}
		__syncthreads();
}

/**
 * TODO:
 *     select  candidates with small distance from array of groupQuerySet_candidates
 *
 *     note:
 *     1. the number of candidates for selection should be able to cached in shared memory
 *
 *     2. there are two cases for select the candidates, controlled by "candidates_number_fixed"
 * 			1): the number of candidates is fixed number, but sometimes the number of candidates may be less than the fixed number,the actually
 * 			    numer of candidates is stored in "groupQuerySet_candidates_idxInfo", the default candidate number is stored in "candidats_number_fixed"
 *
 * 			2): the number of candidates is flexible, the end position of the candidates for each query item is stored in
 * 			    "groupQuerySet_candidates_idxInfo". In this case, candidats_number_fixed is set as 0
 */
__global__ void select_candidates_mergeSort(
		const int select_num,
		const int candidates_number_fixed,//input, please refer to note 2
		const int max_candidate_num,//record the maximum number of candidates to be selected in groupQuerySet_candidates
		CandidateEntry* groupQuerySet_candidates, //input: candidates
		const int* groupQuerySet_Candidates_idxInfo, //record the idx for each group query in d_groupQuerySet_Candidates
		CandidateEntry* groupQuerySet_candidates_selected, //output: selected result are stored here
		int * groupQuerySet_candidates_selected_size //output: record the size, note this is not the idx, only record the size
		) {

	extern __shared__ CandidateEntry sm_array[];

	CandidateEntry* candidates_array = sm_array;
	CandidateEntry* temp_array = &sm_array[max_candidate_num]; //


	uint candidates_start,candidates_size;
	if(candidates_number_fixed == 0){//the candidates number is flexible, calculate the starting position from prefix-count array
		candidates_start =
			(blockIdx.x == 0) ? 0 : (groupQuerySet_Candidates_idxInfo[blockIdx.x - 1]);
		candidates_size = groupQuerySet_Candidates_idxInfo[blockIdx.x] - candidates_start;
	}else{//the candidates number is fixed, load the actually number of the candidates
		candidates_start = blockIdx.x * candidates_number_fixed;
		candidates_size = groupQuerySet_Candidates_idxInfo[blockIdx.x];
	}

	int round_compDist = (candidates_size) / blockDim.x
			+ ((candidates_size) % blockDim.x != 0); //scan every elements,

	//load data into shared memory
	for (uint i = 0; i < round_compDist; i++) {
		uint index = i * blockDim.x + threadIdx.x;
		if (index < candidates_size) {
			candidates_array[index] = groupQuerySet_candidates[candidates_start	+ index];
			//temp_array[index] = 0;
		}

	}
	__syncthreads();
	//only select the first select_num elements use sort merge
	if (candidates_size > select_num) {//
		select_candidateEntry_mergeSort(candidates_array, temp_array,
				candidates_size, select_num, true);
	} else { //if smaller than select_num, sort all of them
		blk_sort_candidateEntry(candidates_array, temp_array, candidates_size);
	}

	uint result_start = blockIdx.x * select_num;//position to storethe results
	uint round_result = (select_num) / blockDim.x
			+ ((select_num) % blockDim.x != 0);

	//scan every elements,
	for (uint i = 0; i < round_result; i++) {
		uint index = i * blockDim.x + threadIdx.x;

		if (index < select_num && index < candidates_size) {

			groupQuerySet_candidates_selected[result_start + index] =
					candidates_array[index];

			if (index == 0) {

				groupQuerySet_candidates_selected_size[blockIdx.x] =
						candidates_size > select_num ? select_num : candidates_size;

			}
		}
	}
}

/**
 * TODO:
 *     add new results in to groupQuerySet_candidates_topk and update the threshold for topk query.
 *
 *     note:
 *     1. For every query item (of every group query), there should be k candidates in the result heap groupQuerySet_candidates_topk.
 *        Therefore, there is no need to maintain the groupQuerySet_candidates_topk_size
 *     2. four times of the (top)k candidates should be able to cached in the shared memory
 *
 *
 */
__global__ void maintain_topkCandidates_mergeSort(
	const int topk,
	CandidateEntry* groupQuerySet_candidates_topk,//input and output: there must be k result, no need for int to count the address
	CandidateEntry* groupQuerySet_candidates,//input: result are stored here
	int * groupQuerySet_candidates_size, //note this is size, not address, each query occupied k items size, but some be padded with NULL entry
	float* groupQuerySet_threshold//output: update the threshold for every query item for every group query
	){

	extern __shared__ CandidateEntry sm_array[];//four times of topk

	CandidateEntry* candidates_array = sm_array;
	CandidateEntry* temp_array = &sm_array[2*topk];//to store the temp result, allocate for merge sort

	//every query occupy topk candidates,
	//note groupQuerySet_candidates_topk and groupQuerySet_candidates have the same starting position in their own arrays respectively
	uint candidates_start = blockIdx.x*topk;

	uint new_candidates_size = groupQuerySet_candidates_size[blockIdx.x];//should be smaller than topk
	new_candidates_size = new_candidates_size<topk ? new_candidates_size:topk;


	int round = topk/blockDim.x + (topk%blockDim.x!=0);
	//load data into shared memory
	for(int i=0;i<round;i++){
		uint index = i*blockDim.x + threadIdx.x;

		if(index<topk){
			candidates_array[index] = groupQuerySet_candidates_topk[candidates_start+index];
		}
		if(index<new_candidates_size){
			candidates_array[index+topk] = groupQuerySet_candidates[candidates_start+index];

		}


	}
	 __syncthreads();

	 //improve here !!! which is faster?
	blk_sort_candidateEntry(candidates_array, temp_array, topk+new_candidates_size);
	//select_candidateEntry_mergeSort(candidates_array,temp_array,topk+new_candidates_size,topk, true);

	//output
	for(int i=0;i<round;i++){
		uint index = i*blockDim.x + threadIdx.x;
		if(index<topk){
			groupQuerySet_candidates_topk[candidates_start+index] = candidates_array[index];
		}
	}

	if(threadIdx.x == 0){
		groupQuerySet_threshold[blockIdx.x] = candidates_array[topk-1].dist;
	}


}

//__global__ void select_candidates_random(
/**
 * TODO:
 * randomly select "candidates_num" from d_groupQuerySet_candidates to groupQuerySet_candidates_select
 *
 * one query item for one block
 */
__global__ void select_candidates_random(
		const int selectedCandidates_num,
		CandidateEntry* d_groupQuerySet_candidates, //input:
		const int* d_groupQuerySet_Candidates_endIdx, //record the idx for each group query in d_groupQuerySet_Candidates
		CandidateEntry* groupQuerySet_candidates_selected, //output: result are stored here
		int* d_groupQuerySet_candidates_selected_size//output: note: this is not endIdx, do not do inclusive sum, we only count the number of candidates
		) {

	uint candidates_start = (blockIdx.x == 0) ? 0 : (d_groupQuerySet_Candidates_endIdx[blockIdx.x - 1]);
	uint candidates_end = d_groupQuerySet_Candidates_endIdx[blockIdx.x];
	uint candidates_size = candidates_end - candidates_start;
	if(candidates_size ==0 ) return;//this query item does not have any candidates

	d_groupQuerySet_candidates_selected_size[blockIdx.x] = candidates_size<=selectedCandidates_num? candidates_size:selectedCandidates_num;

	uint result_start = blockIdx.x * selectedCandidates_num;
	uint round_result = (selectedCandidates_num) / blockDim.x + ((selectedCandidates_num) % blockDim.x != 0); //scan every elements,

	uint candidate_interval = (candidates_size/selectedCandidates_num == 0) ? 1 : (candidates_size/selectedCandidates_num);

	for (uint i = 0; i < round_result; i++) {

		uint index = i * blockDim.x + threadIdx.x;

		if (index < selectedCandidates_num&&index<candidates_size) {
			groupQuerySet_candidates_selected[result_start + index] =
					d_groupQuerySet_candidates[candidates_start+index*candidate_interval];

			//groupQuerySet_candidates[index*candidate_interval+random()%candidate_interval];//improve here !!! not random selected
		}
	}
}




//the blocks is equal to the number of group queries
__global__ void check_groupQuery_allItemsFinished(
		GroupQuery_info** d_groupQuery_info_set, //input:group query
		bool* groupQuery_unfinished//output: the status of every group queries
		){

	extern __shared__ bool groupQuery_itemsFinished[];//the shared memory size is equal to the number of items in a group query

	int round = (dev_item_num_perGroupQuery[0]/blockDim.x)+(dev_item_num_perGroupQuery[0]%blockDim.x !=0);

	for(int i=0;i<round;i++){

		uint index = i*blockDim.x+threadIdx.x;
		if(index<dev_item_num_perGroupQuery[0]){
			groupQuery_itemsFinished[index] = d_groupQuery_info_set[blockIdx.x]->depressed_item_query_finished[index];

		}

	}


	__syncthreads();
	uint pre_s = dev_item_num_perGroupQuery[0];
	for (unsigned int s = dev_item_num_perGroupQuery[0] / 2; s > 0; s >>= 1) {

		for(int i=0;i<round;i++){
			uint index =  i*blockDim.x+threadIdx.x;
			if (index < s) {
				groupQuery_itemsFinished[index] &= groupQuery_itemsFinished[index + s];
			}
		}

		if(pre_s%2 == 1 && 0 == threadIdx.x ) {
			groupQuery_itemsFinished[0] &= groupQuery_itemsFinished[2*s];
		}
		pre_s = s;

	}
	__syncthreads();
	if(threadIdx.x == 0){
		groupQuery_unfinished[blockIdx.x] = !(groupQuery_itemsFinished[0]);
	}

}



//junk code:


