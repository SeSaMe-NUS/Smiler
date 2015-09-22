#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h> 
#include <math.h>
#include <assert.h>
#include <set>
#include <algorithm>

#include <device_launch_parameters.h>

#include <cuda.h>

#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include "generalization.h"

using namespace std;
using namespace thrust;





/* GPU device constants */


// TODO: need to optimize it using __constant__ memory
//__device__ device_vector<int> dev_minDomainForAllDimension;
//__device__ device_vector<int> dev_maxDomainForAllDimension;

//GPU code
__device__ __constant__ int dev_dimensionNum_perQuery[1];
__device__ __constant__ int dev_numOfDocToExpand[1];
__device__ __constant__ int dev_numOfQuery[1];
__device__ __constant__ int dev_maxFeatureNumber[1];


void initialize_dev_constMem( const InvertListSpecGPU& spec )
{


	printf ("# index dim: %d \n", spec.totalDimension );
	printf ("# numOfDocToExpand: %d \n", spec.numOfDocToExpand );
	printf ("# numOfQuery: %d \n", spec.numOfQuery );
	printf ("# maxFeatureID: %d \n", spec.maxFeatureNumber );


	//HANDLE_ERROR( cudaMemcpyToSymbol( dev_dimensionNum_perQuery, &(spec.totalDimension), sizeof(int), 0, cudaMemcpyHostToDevice ) );//in this case, we consider the full dimension search and query has the same number of dimensions which is equal to the index dimensions
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_numOfDocToExpand, &(spec.numOfDocToExpand), sizeof(int), 0, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_numOfQuery, &(spec.numOfQuery), sizeof(int), 0, cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_maxFeatureNumber, &(spec.maxFeatureNumber), sizeof(int), 0, cudaMemcpyHostToDevice ) );


}

void init_dev_dimNumPerQuery_constMem(int dimensionNum_perQuery){

	printf("# dimension of each query in gpuMananger:%d \n",dimensionNum_perQuery);

	//in this case, we consider the sub dimension search, still assume all query has the same number of dimensions which is equal to the dev_dimensionNum_perQuery
	HANDLE_ERROR( cudaMemcpyToSymbol( dev_dimensionNum_perQuery, &(dimensionNum_perQuery), sizeof(int), 0, cudaMemcpyHostToDevice ) );

}


void freeGPUMemory()
{
//	cudaFree( dev_totalDimension );
//	cudaFree( dev_numOfDocToExpand );
//	cudaFree( dev_numOfQuery );
//	cudaFree( dev_maxFeatureID );
}




/**
 * sort the data_array in shared memory by descending order
 */
__device__ void blk_sort_inSharedMemory(float* data_array, float* temp_array, int size)
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
						temp_array[output_ptr++] = data_array[y_pointer++];
					else if (y_pointer >= end_idx_y)
						temp_array[output_ptr++] = data_array[x_pointer++];
					else if (data_array[x_pointer] < data_array[y_pointer])//key operation
						temp_array[output_ptr++] = data_array[y_pointer++];
					else
						temp_array[output_ptr++] = data_array[x_pointer++];
				}
			}
		}
		__syncthreads();

		//copy back to shared_array
		int round = size / blockDim.x + (size % blockDim.x != 0);
		for (int i = 0; i < round; i++) {
			int idx = i * blockDim.x + threadIdx.x;
			if (idx < size) {
				data_array[idx] = temp_array[idx];
			}
		}
		__syncthreads();
	}
	__syncthreads();
}

/**
 * edit:zhou jingbo
 * TODO:
 * sum the data in data_array
 */

__device__ float  blk_sum_sharedMemory( float* data_array, int size){

	int pre_s = size;
	for (unsigned int s = size / 2; s > 0; s >>= 1) {

		int sum_round = s/blockDim.x+ (s%blockDim.x!=0);
		for(int si=0;si<sum_round;si++){
			int idx = si*blockDim.x+threadIdx.x;
			if (idx < s) {
				data_array[idx] += data_array[idx + s];
			}

		}

		if(pre_s%2 == 1 && 0 == threadIdx.x ) {
			data_array[0] += data_array[2*s];
		}
		pre_s = s;
		__syncthreads();
	}

	float res = data_array[0];
	__syncthreads();
	return res;

}


/**
 * edit:zhou jingbo
 * TODO:
 * aggregation the data in data_array, with
 */
template <class AGG>
__device__ float  blk_aggregation_sharedMemory( float* data_array, int size, AGG agg){

	int pre_s = size;
	for (unsigned int s = size / 2; s > 0; s >>= 1) {

		int sum_round = s/blockDim.x+ (s%blockDim.x!=0);
		for(int si=0;si<sum_round;si++){
			int idx = si*blockDim.x+threadIdx.x;
			if (idx < s) {
				data_array[idx] = agg.op(data_array[idx], data_array[idx + s]);
			}
		}

		if(pre_s%2 == 1 && 0 == threadIdx.x ) {
			data_array[0] = agg.op( data_array[0], data_array[2*s]);
		}
		pre_s = s;
		__syncthreads();
	}
	float res = data_array[0];
	__syncthreads();
	return res;
}



template __device__ float  blk_aggregation_sharedMemory<Blk_Sum>( float* data_array, int size, Blk_Sum bs);
template __device__ float  blk_aggregation_sharedMemory<Blk_Max>( float* data_array, int size, Blk_Max bm);


__global__ void printQueryInfo(WindowQueryInfo** query_list, int size)
{
	if ( threadIdx.x < size )
	{
		WindowQueryInfo* temp = query_list[threadIdx.x];
		temp->print();

//		(*query_list)[threadIdx.x].print();
	}
}

__global__ void  printConstMem(){

	printf("===================printConstMem()=================\n");
	printf ("device constant::print() \n");
	printf ("dev_totalDimension[0]: %d \n", dev_dimensionNum_perQuery[0] );
	printf ("dev_numOfDocToExpand[0]: %d \n", dev_numOfDocToExpand[0] );
	printf ("dev_numOfQuery[0]: %d \n", dev_numOfQuery[0] );
	printf ("dev_maxFeatureID[0]: %d \n", dev_maxFeatureNumber[0] );


}



/**
 * edit:by zhou jingbo
 *     a. add customized distance function, transfer into by struct
 *
 * NOTE : 1.now assume all query has the same number of dimensions, which is stored in *dev_totalDimension
 *
 *
 *  Edit note:
 * 		1. add template to support bucketlized index whit bucket width > 1 (by jingbo)
 *
 *
 */
template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__global__ void compute_mapping_saving_pos_KernelPerDim_template(
		WindowQueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* _query_feature,//output, record cout&ACD table
		bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc) {

	//int bid = blockIdx.x; // mapped to different dimension of a query

	int tid = threadIdx.x;
	int block_size = blockDim.x;

	int qid  =  blockIdx.x/dev_dimensionNum_perQuery[0];
	int dimId= blockIdx.x%dev_dimensionNum_perQuery[0]; //mapped to different dimension index of A QUERY (note that: this dimId is to index the query dimension, i.e. by query->searchDim[dimId] to locate the true dimensions
	WindowQueryInfo* query = query_list[qid];

	if (dimId < query->numOfDimensionToSearch) {
		//for(int iter = 0; iter < query->numOfDimensionToSearch; iter++){

		float dim_weight = query->depressed_dimWeight[dimId];
		float dim_dist_func = query->depressed_distanceFunc[dimId];
		float dim_lb_dist_func = query->lowerBoundDist[dimId];
		float dim_ub_dist_func = query->upperBoundDist[dimId];
		int bound_down_search = query->depressed_lowerBoundSearch[dimId];
		int bound_up_search = query->depressed_upperBoundSearch[dimId];

		int down_pos = query->depressed_lastPos[dimId].x;
		int up_pos = query->depressed_lastPos[dimId].y;
		float data_keyword = query->keyword[dimId];
		int dim = query->depressed_searchDim[dimId];
		GpuIndexDimensionEntry indexDimEntry = indexDimensionEntry_vec[dim];

		int index_dim_value = keywordMap.mapping(data_keyword,indexDimEntry.bucketWidth);




		//int keyword = y_dim_value + dim * MAX_DIM_VALUE;
		int keyword_indexMapping = index_dim_value + dim * max_value_per_dimension;


		if (down_pos == 0 && up_pos == 0) {
			int invert_list_start =
					keyword_indexMapping == 0 ? 0 : invert_list_idx[keyword_indexMapping - 1];
			int invert_list_end = invert_list_idx[keyword_indexMapping];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureNumber[0]+inv_ent;
					//query_feature[target_idx].count += 1;
					atomicAdd(&(_query_feature[target_idx].count), 1);

				}
			}
		}


		if (point_search) {
			//__syncthreads();
			//continue;
			return;
		}



		// going downward
		//int down_compute = invert_list_spec.numOfDocToExpand;
		int down_compute = (*dev_numOfDocToExpand);
		int index_dim_down_value = index_dim_value - down_pos; // move the position that last iteartion possessed.


		while (index_dim_down_value - 1 >= indexDimEntry.minDomain // make sure dimension is above minimum dimension value
		&& down_compute > 0	// make sure the number of compute element is above 0
		&& index_dim_down_value >= index_dim_value - bound_down_search) {
			index_dim_down_value--;
			//int down_keyword = down_value + dim * MAX_DIM_VALUE;
			int down_keyword = index_dim_down_value + dim * max_value_per_dimension;

			int invert_list_start =
					down_keyword == 0 ? 0 : invert_list_idx[down_keyword - 1];
			int invert_list_end = invert_list_idx[down_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float data_down_value = lastPosMap.map_indexToData_down(index_dim_down_value,indexDimEntry.bucketWidth);//the down value in data space
			float true_dist = distFunc.dist(index_dim_value, data_down_value,
					dim_dist_func, dim_lb_dist_func, dim_weight);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureNumber[0]+inv_ent;
					//int target_idx = bid*invert_list_spec.maxFeatureID+inv_ent;
					//query_feature[target_idx].count += 1;
					//query_feature[target_idx].ACD += true_dist;
					atomicAdd(&(_query_feature[target_idx].count), 1);
					atomicAdd(&(_query_feature[target_idx].ACD), true_dist);
					//if(bid == 1) printf("query %d meet doc %d going down\n",bid,inv_ent);

				}
			}

			down_compute -= invert_list_size;
		}

		// going upward
		//int up_compute = invert_list_spec.numOfDocToExpand;
		int up_compute = (*dev_numOfDocToExpand);
		int index_dim_up_value = index_dim_value + up_pos; // move the position that last iteartion possessed.


		while (index_dim_up_value + 1 <= indexDimEntry.maxDomain // make sure dimension is below maximum dimension value
		&& up_compute > 0 // make sure the number of compute element is above 0
		&& index_dim_up_value <= index_dim_value + bound_up_search) {
			index_dim_up_value++;
			//int up_keyword = up_value + dim * MAX_DIM_VALUE;
			int up_keyword = index_dim_up_value + dim * max_value_per_dimension;

			int invert_list_start =
					up_keyword == 0 ? 0 : invert_list_idx[up_keyword - 1];
			int invert_list_end = invert_list_idx[up_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float data_up_value = lastPosMap.map_indexToData_up((index_dim_up_value),indexDimEntry.bucketWidth);
			float true_dist = distFunc.dist(index_dim_value, data_up_value,
					dim_dist_func, dim_ub_dist_func, dim_weight);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureNumber[0]+inv_ent;
					/*int target_idx = bid*invert_list_spec.maxFeatureID+inv_ent;
					 query_feature[target_idx].count += 1;
					 query_feature[target_idx].ACD += true_dist;*/
					atomicAdd(&(_query_feature[target_idx].count), 1);
					atomicAdd(&(_query_feature[target_idx].ACD), true_dist);
					//if(bid == 1) printf("query %d meet doc %d going up\n",bid,inv_ent);
				}
			}

			up_compute -= invert_list_size;
		}

		__syncthreads();
		if (tid == 0) {
			query->depressed_lastPos[dimId].x = index_dim_value - index_dim_down_value;
			query->depressed_lastPos[dimId].y = index_dim_up_value - index_dim_value;
			//printf("Dimension %d with dim value %d is up value: %d and down value %d\n",bid,y_dim_value,query->lastPos[dim].x,query->lastPos[dim].y);
		}
	}

}

template __global__ void compute_mapping_saving_pos_KernelPerDim_template<DataToIndex_keywordMap_bucketUnit,IndexToData_lastPosMap_bucketUnit, Lp_distance>(
		WindowQueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		DataToIndex_keywordMap_bucketUnit intListmap,
		IndexToData_lastPosMap_bucketUnit lastPosMap,
		Lp_distance inLpDist);

template __global__ void compute_mapping_saving_pos_KernelPerDim_template<DataToIndex_keywordMap_bucket,IndexToData_lastPosMap_bucket_exclusive, Lp_distance>(
		WindowQueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		DataToIndex_keywordMap_bucket intListmap,
		IndexToData_lastPosMap_bucket_exclusive lastPosMap,
		Lp_distance inLpDist);


//==============================following part: written for one kernel takes one query, some code can be reused


/**
 * //compute how many results for each query
// count_vec is for the output position for each thread
// blk_count_vec is for the later processing for selecting topK document for each query since each block correspond to a query
//this is v2 version which is consistent with GPUManager::bi_direction_query_KernelPerQuery_V2() function
//refine: remove atomicAdd, create shared memory for buffer the count result
 */
__global__ void prefix_count_KernelPerQuery(
		QueryFeatureEnt* query_feature,//d_query_feature.data(), this is count&ACD table
		int* count_vec, //threshold_count.data()
		int* blk_count_vec,//query_result_count.data()
		int threshold
		)
{
	int bid = blockIdx.x; //block number is: invert_list_spec_host.numOfQuery
	int tid = threadIdx.x; //thread number per block is: THREAD_PER_BLK
	int oid = bid * blockDim.x + tid;

	int block_start = bid * (dev_maxFeatureNumber[0]);
	int block_end = block_start + (dev_maxFeatureNumber[0]);
	int round = (dev_maxFeatureNumber[0]) / blockDim.x + ( (dev_maxFeatureNumber[0]) % blockDim.x != 0);


	extern __shared__ int sm_countVec[];//shared memeory dynamic alloc with the size of blockDim.x

	int count = 0;
	for(int i = 0; i < round; i++){
		int idx = block_start + i * blockDim.x + tid;
		if(idx < block_end){
			if(query_feature[idx].count > threshold){
				count++;
			}
		}
	}

	sm_countVec[tid] = count_vec[oid] = count;
	__syncthreads();
	//note: the size of sm_countVec is just the number of threads per block
    for(unsigned int s = blockDim.x/2;s>0;s>>=1){
    	if(tid<s){
    		sm_countVec[tid] +=sm_countVec[tid+s];
    	}
    }
    __syncthreads();

	if(tid == 0) {	blk_count_vec[bid] = sm_countVec[tid];}

}


// this function is used for point search query
__global__ void output_result_point_search(
		QueryFeatureEnt* query_feature,
		int* ending_idx,
		Result* result_vec,
		int threshold)
{
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int start_idx = pos == 0 ? 0 : ending_idx[pos-1];

	int block_start = blockIdx.x * (*dev_maxFeatureNumber);
	int block_end = block_start + (*dev_maxFeatureNumber);
	int round = (*dev_maxFeatureNumber) / blockDim.x + ( (*dev_maxFeatureNumber) % blockDim.x != 0);

	for(int i = 0; i < round; i++){
		int idx = block_start + i * blockDim.x + threadIdx.x;
		if(idx < block_end){
			float count = query_feature[idx].count;
			if(count > threshold){
				Result new_result;
				new_result.query = blockIdx.x;
				new_result.feature_id = i * blockDim.x + threadIdx.x;
				new_result.count = count;
				result_vec[start_idx] = new_result;
				start_idx++;
			}
		}
	}
}




/**
 * author: zhou jignbo
 * TODO:
 * compute the lower bound and upper bound for function output_result_bidrection_search_KernelPerQuery
 *
 * * Edit note:
 * 	1.add template to support bucketlized index whit bucket width > 1 (by jingbo)
 */
template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__device__ void blk_get_eachQueryDimensionBound_template(WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound, float* _queryUpperBound,
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc
		){

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
			//float min_bound_dist_func;
			//float max_bound_dist_func;
			int dim = queryInfo->depressed_searchDim[idx];
			GpuIndexDimensionEntry indexDimEntry = indexDimensionEntry_vec[dim];
			int index_dim_keyword = keywordMap.mapping(data_dim_keyword,indexDimEntry.bucketWidth);

			_queryLowerBound[idx] = 0.;
			_queryUpperBound[idx] = 0.;

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

			float data_domain_up_value = lastPosMap.map_indexToData_up(indexDimEntry.maxDomain , indexDimEntry.bucketWidth);
			float data_domain_down_value = lastPosMap.map_indexToData_up(indexDimEntry.minDomain, indexDimEntry.bucketWidth);

			float queryUpperBound_domain_up = distFunc.dist(data_dim_keyword, data_domain_up_value, dim_dist_func_lpType,
					data_dist_upperBound, dim_weight);
			float queryUpperBound_domain_down = distFunc.dist(data_dim_keyword, data_domain_down_value, dim_dist_func_lpType,
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

			_queryUpperBound[idx] = fmaxf(queryUpperBound_domain_up,queryUpperBound_domain_down);
		}

	}

	__syncthreads();
}


template __device__ void blk_get_eachQueryDimensionBound_template<DataToIndex_keywordMap_bucketUnit,IndexToData_lastPosMap_bucketUnit, Lp_distance>(
		WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound, float* _queryUpperBound,
		DataToIndex_keywordMap_bucketUnit intListmap,
		IndexToData_lastPosMap_bucketUnit lastPosMap,
		Lp_distance inLpDist
		);


template __device__ void blk_get_eachQueryDimensionBound_template<DataToIndex_keywordMap_bucket,IndexToData_lastPosMap_bucket_inclusive, Lp_distance>(
		WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound, float* _queryUpperBound,
		DataToIndex_keywordMap_bucket intListmap,
		IndexToData_lastPosMap_bucket_inclusive lastPosMap,
		Lp_distance inLpDist
		);

/**
 * edit zyx, and jingbo
 * TODO : apply the distance in the parameter
 *
 * this function is used for bi directional search query, which is consistent and called before prefix_count_KernelPerQuery()
 *
 * Edit note:
 * 	1.add template to support bucketlized index whit bucket width > 1 (by jingbo)
 */
//
template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__global__ void output_result_bidrection_search_KernelPerQuery_template(
	QueryFeatureEnt* query_feature,
	WindowQueryInfo** query_set, // remeber the expansion position for each query
	int* ending_idx, //threshold_count
	Result* result_vec,//record the lower bound and upper bound
	int threshold,
	GpuIndexDimensionEntry* indexDimensionEntry_vec,
	KEYWORDMAP keywordMap,
	LASTPOSMAP lastPosMap,
	DISTFUNC distFunc
	)
{

	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	int start_idx = pos == 0 ? 0 : ending_idx[pos-1];

	int block_start = blockIdx.x * (*dev_maxFeatureNumber);
	int block_end = block_start + (*dev_maxFeatureNumber);
	int round = (*dev_maxFeatureNumber) / blockDim.x + ((*dev_maxFeatureNumber) % blockDim.x != 0);

	WindowQueryInfo* queryInfo = query_set[blockIdx.x];


	int block_num_search_dim = queryInfo->numOfDimensionToSearch;


	// dynamic allocate shared memory. specify outside in kenel calls
	extern __shared__ float queryBoundMem[];

	float* queryLowerBound = queryBoundMem;
	float* queryUpperBound = &queryBoundMem[dev_dimensionNum_perQuery[0]];
	float* temp_shared = &queryBoundMem[2*dev_dimensionNum_perQuery[0]];


	blk_get_eachQueryDimensionBound_template(queryInfo, block_num_search_dim,  indexDimensionEntry_vec,
				queryLowerBound,  queryUpperBound,
				 keywordMap, lastPosMap, distFunc);



	// sort the lower and up bound
	blk_sort_inSharedMemory(queryLowerBound, temp_shared, block_num_search_dim);

	float lower_bound_sum = blk_sum_sharedMemory( temp_shared,  block_num_search_dim);//queryLowerBound is still stored in temp_shared
	__syncthreads();//note: this is important, let all threads get the return value before next step

	blk_sort_inSharedMemory(queryUpperBound, temp_shared, block_num_search_dim);//



	for(int i = 0; i < round; i++)
	{
		int idx = block_start + i * blockDim.x + threadIdx.x;
		if(idx < block_end)
		{
			int count = query_feature[idx].count;

			if(count > threshold)
			{
				Result new_result;
				new_result.query = blockIdx.x;
				new_result.feature_id = i * blockDim.x + threadIdx.x;
				new_result.count = count;

				float ACD = query_feature[idx].ACD;

				// DL(f) = DL(us) - sum(max count number of dt) + ACD(f)
				float lb = lower_bound_sum;
				for(int m = 0; m < count; m++)
				{

					lb -= queryLowerBound[m];

				}
				lb += ACD;

				// DU(f) = ACD(f) + sum(max 128 - c of upper bound)
				float ub = ACD;
				for(int m = 0; m < block_num_search_dim - count; m++)
				{
					ub += queryUpperBound[m];
				}


				new_result.lb = lb;
				new_result.ub = ub;

				result_vec[start_idx] = new_result;
				start_idx++;

			}

		}
	}
}


template __global__ void output_result_bidrection_search_KernelPerQuery_template<DataToIndex_keywordMap_bucketUnit,IndexToData_lastPosMap_bucketUnit, Lp_distance>(
	QueryFeatureEnt* query_feature,
	WindowQueryInfo** query_set, // remeber the expansion position for each query
	int* ending_idx, //threshold_count
	Result* result_vec,//record the lower bound and upper bound
	int threshold,
	GpuIndexDimensionEntry* indexDimensionEntry_vec,
	DataToIndex_keywordMap_bucketUnit intListmap,
	IndexToData_lastPosMap_bucketUnit lastPosMap,
	Lp_distance inLpDist
	);




/**TODO:
 * sort the data into multiple partitions (each size of partition is large or equal to initial value of _sg_len, and the final length is write back into _sg_len);
 *
 * Note: the auxiliary function of blk_k_selection_bySortMerge()
 */
__device__ void inline blk_sort_segment(Result* shared_array, Result* temp_array, int size, int& _sg_len, bool sort_ub)
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
					temp_array[output_ptr++] = shared_array[y_pointer++];
				else if (y_pointer >= end_idx_y)
					temp_array[output_ptr++] = shared_array[x_pointer++];
				else if(sort_ub){//select the smallest one, sorting the lower bound with ascending order
				if (shared_array[x_pointer].ub > shared_array[y_pointer].ub)
					temp_array[output_ptr++] = shared_array[y_pointer++];
				else
					temp_array[output_ptr++] = shared_array[x_pointer++];
				}else{ //select the largest one, sorting the upp bound with ascending order
					if (shared_array[x_pointer].lb > shared_array[y_pointer].lb)
						temp_array[output_ptr++] = shared_array[y_pointer++];
					else
						temp_array[output_ptr++] = shared_array[x_pointer++];
				}
			}
		}
		}
		__syncthreads();


		int round = size/blockDim.x + (size % blockDim.x  != 0);
		for(int i = 0; i < round; i++){
			int idx = i * blockDim.x  + threadIdx.x;
			if(idx<size){
			shared_array[idx] = temp_array[idx];
			}
		}

		//__syncthreads();
		//base *= 2;
	}
	_sg_len =base;
}

/**
 *TODO:
 *copy from A to B
 *
 *Note: the auxiliary function of blk_k_selection_bySortMerge()
 *
 */
__device__ void inline blk_cp(Result* A, Result *B, int size){

	int round = size/blockDim.x + (size % blockDim.x  != 0);
		for(int i = 0; i < round; i++){
			int idx = i * blockDim.x  + threadIdx.x;
			if(idx<size){
			B[idx] = A[idx];
			}
		}

}

/**
 * TODO:
 * move the first k item of each partition(with size of base) of A into the ahead of array _B, and the rest of A into the rest of B
 *
 * Note: the auxiliary function of blk_k_selection_bySortMerge()
 */
__device__ void inline blk_k_compact(Result* A, Result* _B, int& compact_size, const int size, const int k, int base){
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

/**
 * TODO:
 * select the first k elements with smallest Result (if sort_ub = true, sort with Result.ub, else with Result.lb) from data_array
 *
 * NOTE: important function, which is auxiliary function of terminateCheck_kSelection_KernelPerQuery()
 */

#define MAX_SHARED_RESULT_MEM 500
__device__ void blk_k_selection_bySortMerge(Result* data_array, Result* temp_array, const int size,  const int k, bool sort_ub)
{

	//create share memeory
	__shared__ Result share_temp[MAX_SHARED_RESULT_MEM];

	int sg_len = k;
	int compact_size=0;
	if(size<MAX_SHARED_RESULT_MEM){
			temp_array = &share_temp[0];
	}

	blk_sort_segment(data_array, temp_array, size, sg_len,sort_ub);
	blk_k_compact(data_array, temp_array, compact_size, size, k, sg_len);	//compact to temp_array
	blk_cp(temp_array, data_array, size);

	int num = size / sg_len + (size % sg_len != 0);

	for (; num > 1; num =(num/2+num%2)) {
		if(compact_size<MAX_SHARED_RESULT_MEM){
			temp_array = &share_temp[0];
		}
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
					temp_array[output_ptr++] = data_array[y_pointer++];
				else if (y_pointer >= end_idx_y)
					temp_array[output_ptr++] = data_array[x_pointer++];
				else if(sort_ub){////select the smallest one, sorting the lower bound with ascending order
					if (data_array[x_pointer].ub > data_array[y_pointer].ub)
					temp_array[output_ptr++] = data_array[y_pointer++];
					else
					temp_array[output_ptr++] = data_array[x_pointer++];
				} else{//select the smallest one, sorting the upper bound with ascending order
					if (data_array[x_pointer].lb > data_array[y_pointer].lb)
						temp_array[output_ptr++] = data_array[y_pointer++];
					else
						temp_array[output_ptr++] = data_array[x_pointer++];
				}
			}

			while (x_pointer < end_idx_x) {
				temp_array[output_ptr++] = data_array[x_pointer++];
			}
			while (y_pointer < end_idx_y) {
				temp_array[output_ptr++] = data_array[y_pointer++];
			}
		}
		}

			__syncthreads();

			int k_b_len = compact_size;
			blk_k_compact(temp_array, data_array, compact_size, k_b_len, k, 2*k);	//compact to shared_array

	}
		//__syncthreads();
}



/**
 * author: zhou jingbo
 * TODO:  first select the k results with smallest upper bound and the smallest lower bound of the rest results,
 * 		  and then check the termination condition
 *
 *
 * ..note:  is auxiliary function of GPUManager::bi_direction_query_KernelPerDim_V2()
 */
__global__ void terminateCheck_kSelection_KernelPerQuery(
		Result* boundData,
		Result* temp,//d_result_lb_sorted array
		int* end_idx,//query_result_count.data()
		int* output,//d_valid_query.data()
		const int K){

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int blk_start_idx = (bid == 0) ? 0 : end_idx[bid - 1];
	int blk_end_idx = end_idx[bid];

	int interval = blk_end_idx - blk_start_idx;

	if (interval <= K)
		return;


	Result* blk_data = &(boundData[blk_start_idx]);
	Result* blk_temp = &(temp[blk_start_idx]);

	bool sort_ub = true;
	blk_k_selection_bySortMerge(blk_data, blk_temp, interval, K, sort_ub);
	blk_k_selection_bySortMerge(blk_data + K, blk_temp + K, interval-K, 1, !sort_ub);

	__syncthreads();
	if(tid == 0){
		if(blk_data[K].lb >= blk_data[K-1].ub){
			output[bid] = 0;
		}
	}

}

/**
 * TODO:
 * extract the topK result
 */
__global__ void extract_topK_KernelPerQuery ( Result* ub_sorted, int* end_idx, Result* output, int K)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int oid = bid * K + tid;

	int blk_start_idx = bid == 0 ? 0 : end_idx[bid-1];
	int blk_end_idx = blk_start_idx + K;

	int round = K / blockDim.x + (K % blockDim.x != 0);

	for (int i = 0; i < round; i++)
	{
		int idx = blk_start_idx + i * blockDim.x + tid;
		if (idx < blk_end_idx)
		{
			output[oid] = ub_sorted[idx];
			oid += blockDim.x;
		}
	}
}


//=============================================the following code is depressed and should not be used. Finally all the code will be removed from this project
/**
 * TODO:
 * compute how many results for each query
 *
 * // count_vec is for the output position for each thread
// blk_count_vec is for the later processing for selecting topK document for each query since each block correspond to a query
//this is v1 version which is consistent with GPUManager::bi_direction_query_KernelPerQuery_V1() function
 *
 * NOTE: //this function is depressed and should not be used any more, we keep this function with debug purpose and experiment purpose
 */
__global__ void depressed_prefix_count_KernelPerQuery_V1(
		QueryFeatureEnt* query_feature,//d_query_feature.data(), this is count&ACD table
		int* count_vec, //threshold_count.data()
		int* blk_count_vec,//query_result_count.data()
		int threshold,
		int* dev_minDomainForAllDimension,
		int* dev_maxDomainForAllDimension)
{
	int bid = blockIdx.x; //block number is: invert_list_spec_host.numOfQuery
	int tid = threadIdx.x; //thread number per block is: THREAD_PER_BLK
	int oid = bid * blockDim.x + tid;

	int block_start = bid * (dev_maxFeatureNumber[0]);
	int block_end = block_start + (dev_maxFeatureNumber[0]);
	int round = (dev_maxFeatureNumber[0]) / blockDim.x + ( (dev_maxFeatureNumber[0]) % blockDim.x != 0);


	__shared__ int shared_count;
	if(tid == 0) shared_count = 0;
	__syncthreads();

	int count = 0;
	for(int i = 0; i < round; i++){
		int idx = block_start + i * blockDim.x + tid;
		if(idx < block_end){
			if(query_feature[idx].count > threshold){
				count++;
				atomicAdd(&shared_count,1);
			}
		}
	}

	count_vec[oid] = count;
	__syncthreads();



	if(tid == 0) blk_count_vec[bid] = shared_count;
}





//this function is depressed and should not be used, but we keep this function for experiment purpose
__global__ void depressed_compute_mapping_saving_pos_KernelPerDim(
		WindowQueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec
		) {

// mapped to different dimension of a query  	GpuIndexDimensionEntry* indexDimensionEntry_vec

	int tid = threadIdx.x;
	int block_size = blockDim.x;

	int qid  =  blockIdx.x/dev_dimensionNum_perQuery[0];
	int dimId= blockIdx.x%dev_dimensionNum_perQuery[0]; //mapped to different dimension index of A QUERY (note that: this dimId is to index the query dimension, i.e. by query->searchDim[dimId] to locate the true dimensions
	WindowQueryInfo* query = query_list[qid];


	if (dimId < query->numOfDimensionToSearch) {
		//for(int iter = 0; iter < query->numOfDimensionToSearch; iter++){

		float dim_weight = query->depressed_dimWeight[dimId];
		float dim_dist_func = query->depressed_distanceFunc[dimId];
		float dim_lb_dist_func = query->lowerBoundDist[dimId];
		float dim_ub_dist_func = query->upperBoundDist[dimId];
		int bound_down_search = query->depressed_lowerBoundSearch[dimId];
		int bound_up_search = query->depressed_upperBoundSearch[dimId];
		int down_pos = query->depressed_lastPos[dimId].x;
		int up_pos = query->depressed_lastPos[dimId].y;
		int y_dim_value = depressed_rounding(query->keyword[dimId]);

		int dim = query->depressed_searchDim[dimId];
		GpuIndexDimensionEntry indexDimEntry = indexDimensionEntry_vec[dim];
		//int keyword = y_dim_value + dim * MAX_DIM_VALUE;
		int keyword = y_dim_value + dim * max_value_per_dimension;


		if (down_pos == 0 && up_pos == 0) {
			int invert_list_start =
					keyword == 0 ? 0 : invert_list_idx[keyword - 1];
			int invert_list_end = invert_list_idx[keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureNumber[0]+inv_ent;
					//query_feature[target_idx].count += 1;
					atomicAdd(&(query_feature[target_idx].count), 1);

				}
			}
		}


		if (point_search) {
			//__syncthreads();
			//continue;
			return;
		}



		// going downward
		//int down_compute = invert_list_spec.numOfDocToExpand;
		int down_compute = (*dev_numOfDocToExpand);
		int down_value = y_dim_value - down_pos; // move the position that last iteartion possessed.

		while (down_value - 1 >=indexDimEntry.minDomain // // make sure dimension is above minimum dimension value
		&& down_compute > 0	// make sure the number of compute element is above 0
		&& down_value >= y_dim_value - bound_down_search) {
			down_value--;
			//int down_keyword = down_value + dim * MAX_DIM_VALUE;
			int down_keyword = down_value + dim * max_value_per_dimension;

			int invert_list_start =
					down_keyword == 0 ? 0 : invert_list_idx[down_keyword - 1];
			int invert_list_end = invert_list_idx[down_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float true_dist = depressed_distance_func(y_dim_value, down_value,
					dim_dist_func, dim_lb_dist_func, dim_weight);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureNumber[0]+inv_ent;
					//int target_idx = bid*invert_list_spec.maxFeatureID+inv_ent;
					//query_feature[target_idx].count += 1;
					//query_feature[target_idx].ACD += true_dist;
					atomicAdd(&(query_feature[target_idx].count), 1);
					atomicAdd(&(query_feature[target_idx].ACD), true_dist);
					//if(bid == 1) printf("query %d meet doc %d going down\n",bid,inv_ent);

				}
			}

			down_compute -= invert_list_size;
		}

		// going upward
		//int up_compute = invert_list_spec.numOfDocToExpand;
		int up_compute = (*dev_numOfDocToExpand);
		int up_value = y_dim_value + up_pos; // move the position that last iteartion possessed.

		//while(up_value + 1 <= invert_list_spec.maxDomainForAllDimension[dim] // make sure dimension is below maximum dimension value
		while (up_value + 1 <= indexDimEntry.maxDomain// // make sure dimension is below maximum dimension value
		&& up_compute > 0 // make sure the number of compute element is above 0
		&& up_value <= y_dim_value + bound_up_search) {
			up_value++;
			//int up_keyword = up_value + dim * MAX_DIM_VALUE;
			int up_keyword = up_value + dim * max_value_per_dimension;

			int invert_list_start =
					up_keyword == 0 ? 0 : invert_list_idx[up_keyword - 1];
			int invert_list_end = invert_list_idx[up_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float true_dist = depressed_distance_func(y_dim_value, up_value,
					dim_dist_func, dim_ub_dist_func, dim_weight);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureNumber[0]+inv_ent;
					/*int target_idx = bid*invert_list_spec.maxFeatureID+inv_ent;
					 query_feature[target_idx].count += 1;
					 query_feature[target_idx].ACD += true_dist;*/
					atomicAdd(&(query_feature[target_idx].count), 1);
					atomicAdd(&(query_feature[target_idx].ACD), true_dist);
					//if(bid == 1) printf("query %d meet doc %d going up\n",bid,inv_ent);
				}
			}

			up_compute -= invert_list_size;
		}

		__syncthreads();
		if (tid == 0) {
			query->depressed_lastPos[dimId].x = y_dim_value - down_value;
			query->depressed_lastPos[dimId].y = up_value - y_dim_value;
			//printf("Dimension %d with dim value %d is up value: %d and down value %d\n",bid,y_dim_value,query->lastPos[dim].x,query->lastPos[dim].y);
		}
	}

}




/**
 * TODO:
 * check the stop conditions
 * this function is auxiliary function of bi_direction_query_KernelPerQuery_V1,
 *
 * Before calling this function, we first need to sort the result in ub_sort(with result.ub) and lb_sorted (with result.lb)
 *
 * this function is depressed and should not be used
 */
__global__ void depressed_terminateCheck_afterSort_KernelPerQuery(
		Result* ub_sorted,
		Result* lb_sorted,//d_result_lb_sorted array
		int* end_idx,//query_result_count.data()
		int* output,//d_valid_query.data()
		int K,
		int round_num)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int blk_start_idx = bid == 0 ? 0 : end_idx[bid-1];
	int blk_end_idx = end_idx[bid];

	if(blk_end_idx - blk_start_idx <= K) return;

	blk_end_idx = blk_start_idx + K + 1;

	int round = (K+1) / blockDim.x + ((K+1) % blockDim.x != 0);

	__shared__ int termination_check;
	if(tid == 0) termination_check = 0;
	__syncthreads();

	for(int i = 0; i < round; i++){
		int idx = blk_start_idx + i * blockDim.x + tid;
		if(idx < blk_end_idx){
			Result lb_elem = lb_sorted[idx];
			bool duplicated = false; // if element in the lb sorted array existed in the ub sorted array
			for(int j = 0; j < K; j++){
				Result ub_elem = ub_sorted[blk_start_idx + j];
				if(lb_elem.feature_id == ub_elem.feature_id){
					duplicated = true;
					break;
				}
			}

			Result ub_elem = ub_sorted[blk_start_idx + K - 1];
			bool lb_greater_ub = (lb_elem.lb >= ub_elem.ub);


			if(!duplicated && lb_greater_ub){

				atomicAdd(&termination_check,1);
				break;

			}
		}
	}
	__syncthreads();
	if(tid == 0){
		if(termination_check != 0){
			output[bid] = 0;
		}
	}
}



//junk code:

//===============================2014.06.10
/**
 * edit zyx, and jingbo
 * TODO : apply the distance in the parameter
 *
 * this function is used for bi directional search query, which is consistent and called before prefix_count_KernelPerQuery()
 */
//
/*

__global__ void output_result_bidrection_search_KernelPerQuery(
	QueryFeatureEnt* query_feature,
	QueryInfo** query_set, // remeber the expansion position for each query
	int* ending_idx, //threshold_count
	Result* result_vec,//record the lower bound and upper bound
	int threshold,
	GpuIndexDimensionEntry* indexDimensionEntry_vec

	)
{

	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	int start_idx = pos == 0 ? 0 : ending_idx[pos-1];

	int block_start = blockIdx.x * (*dev_maxFeatureID);
	int block_end = block_start + (*dev_maxFeatureID);
	int round = (*dev_maxFeatureID) / blockDim.x + ((*dev_maxFeatureID) % blockDim.x != 0);

	QueryInfo* queryInfo = query_set[blockIdx.x];


	int block_num_search_dim = queryInfo->numOfDimensionToSearch;


	// dynamic allocate shared memory. specify outside in kenel calls
	extern __shared__ float queryBoundMem[];

	float* queryLowerBound = queryBoundMem;
	float* queryUpperBound = &queryBoundMem[dev_dimensionNum_perQuery[0]];
	float* temp_shared = &queryBoundMem[2*dev_dimensionNum_perQuery[0]];



	 blk_get_eachQueryDimensionBound(queryInfo, block_num_search_dim,  indexDimensionEntry_vec,
				queryLowerBound,  queryUpperBound );




	// sort the lower and up bound
	blk_sort_inSharedMemory(queryLowerBound, temp_shared, block_num_search_dim);
	float lower_bound_sum = blk_sum_sharedMemory( temp_shared,  block_num_search_dim);//queryLowerBound is still stored in temp_shared

	blk_sort_inSharedMemory(queryUpperBound, temp_shared, block_num_search_dim);//



	for(int i = 0; i < round; i++)
	{
		int idx = block_start + i * blockDim.x + threadIdx.x;
		if(idx < block_end)
		{
			int count = query_feature[idx].count;

			if(count > threshold)
			{
				Result new_result;
				new_result.query = blockIdx.x;
				new_result.feature_id = i * blockDim.x + threadIdx.x;
				new_result.count = count;

				float ACD = query_feature[idx].ACD;

				// DL(f) = DL(us) - sum(max count number of dt) + ACD(f)
				float lb = lower_bound_sum;
				for(int m = 0; m < count; m++)
				{

					lb -= queryLowerBound[m];

				}
				lb += ACD;

				// DU(f) = ACD(f) + sum(max 128 - c of upper bound)
				float ub = ACD;
				for(int m = 0; m < block_num_search_dim - count; m++)
				{
					ub += queryUpperBound[m];
				}


				new_result.lb = lb;
				new_result.ub = ub;

				result_vec[start_idx] = new_result;
				start_idx++;

			}

		}
	}
}
*/


/*

*
 * author: zhou jignbo
 * TODO:
 * compute the lower bound and upper bound for function output_result_bidrection_search_KernelPerQuery


__device__ void blk_get_eachQueryDimensionBound(QueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound, float* _queryUpperBound
		){

	int round = (block_num_search_dim) / blockDim.x
			+ ((block_num_search_dim) % blockDim.x != 0);
	for (int i = 0; i < round; i++) {
		int idx = i * blockDim.x + threadIdx.x;
		if (idx < block_num_search_dim) {

			int data_dim_keyword = queryInfo->keyword[idx];
			float dim_dist_func_lpType = queryInfo->distanceFunc[idx];
			float dim_weight = queryInfo->dimWeight[idx];
			float data_dist_lowerBound = queryInfo->lowerBoundDist[idx];//in data space
			float data_dist_upperBound = queryInfo->upperBoundDist[idx];//in data space
			int index_dim_upperBoundSearch = queryInfo->upperBoundSearch[idx];// search bound when going up, in index space
			int index_dim_lowerBoundSearch = queryInfo->lowerBoundSearch[idx];// search boudn when going down, in index space
			//float min_bound_dist_func;
			//float max_bound_dist_func;
			int dim = queryInfo->searchDim[idx];
			GpuIndexDimensionEntry indexDimEntry = indexDimensionEntry_vec[dim];
			int index_dim_keyword = rounding(data_dim_keyword);

			_queryLowerBound[idx] = 0.;
			_queryUpperBound[idx] = 0.;

			int2 q_pos = queryInfo->lastPos[idx];
			// make sure the bound is correct when upward and downward search all reach the maximum, comment by jingbo
			// modified to the min compare to modified.cu which use max

			bool isReachMin = (q_pos.x >= index_dim_lowerBoundSearch);//(q_pos.x	== (query_dim_value - indexDimensionEntry_vec[dim]));
			bool isReachMax = (q_pos.y >= index_dim_upperBoundSearch);//(q_pos.y == (dev_maxDomainForAllDimension[dim] - query_dim_value));



			int index_reach_up = index_dim_keyword + q_pos.y;
			int index_reach_down = index_dim_keyword - q_pos.x;
			float data_reach_up_value = index_reach_up	* indexDimEntry.bucketWidth;
			float data_reach_down_value = (index_reach_down)	* indexDimEntry.bucketWidth;

			float queryLowerBound_reach_up = distance_func(data_dim_keyword, data_reach_up_value, dim_dist_func_lpType,
					data_dist_upperBound, dim_weight);
			float queryLowerBound_reach_down = distance_func(data_dim_keyword, data_reach_down_value, dim_dist_func_lpType,
					data_dist_lowerBound, dim_weight);

			float data_domain_up_value = (indexDimEntry.maxDomain + 1) 	* indexDimEntry.bucketWidth;
			float data_domain_down_value = (indexDimEntry.minDomain)  * indexDimEntry.bucketWidth;

			float queryUpperBound_domain_up = distance_func(data_dim_keyword, data_domain_up_value, dim_dist_func_lpType,
					data_dist_upperBound, dim_weight);
			float queryUpperBound_domain_down = distance_func(data_dim_keyword, data_domain_down_value, dim_dist_func_lpType,
					data_dist_lowerBound, dim_weight);

			if (isReachMin && isReachMax) {
				//min_lb = max(q_pos.x, q_pos.y);
				//min_lb = (q_pos.x - data_dist_lowerBound) >= (q_pos.y - data_dist_upperBound) ? 	q_pos.x : q_pos.y;
				//min_lb_bound = (q_pos.x - data_dist_lowerBound) >= (q_pos.y - data_dist_upperBound) ? data_dist_lowerBound : data_dist_upperBound;

				_queryLowerBound[idx] = fmaxf(queryLowerBound_reach_up,queryLowerBound_reach_down);


			} else if (isReachMin) {


				_queryLowerBound[idx] = queryLowerBound_reach_up;


			} else if (isReachMax) {

				_queryLowerBound[idx] = queryLowerBound_reach_down;



			} else {


				_queryLowerBound[idx] = fminf(queryLowerBound_reach_up,queryLowerBound_reach_down);
			}







			_queryUpperBound[idx] = fmaxf(queryUpperBound_domain_up,queryUpperBound_domain_down);
		}

	}

	__syncthreads();
}

*/

/**
 * edit zyx, jingbo
 * TODO : apply the distance in the parameter
 *
 * each query take one block (kernel), therefore this function is query oriented parallel computing, this is an old version
 *
 */
/*

// computing the counting while saving the position
__global__ void compute_mapping_saving_pos_KernelPerQuery(
		QueryInfo** query_list,
		InvlistEnt* invert_list,
		int* invert_list_idx,
		QueryFeatureEnt* query_feature,
		bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec

		)
{


	int bid = blockIdx.x; // correspond to query
	int tid = threadIdx.x;
	int block_size = blockDim.x;

	QueryInfo* query = query_list[bid];

	// dynamic allocate shared memory. specify outside in kernel calls
	extern __shared__ int2 pos[];

	if (tid < dev_dimensionNum_perQuery[0])
	{
		pos[tid] = query->lastPos[tid];
	}


	__syncthreads();

	for(int iter = 0; iter < query->numOfDimensionToSearch; iter++)
	{

		float dim_weight = query->dimWeight[iter];
		float dim_dist_func = query->distanceFunc[iter];
		float dim_lb_dist_func = query->lowerBoundDist[iter];
		float dim_ub_dist_func = query->upperBoundDist[iter];
		int bound_down_search = query->lowerBoundSearch[iter];
		int bound_up_search = query->upperBoundSearch[iter];
		int down_pos = pos[iter].x;
		int up_pos = pos[iter].y;


		int y_dim_value = rounding(query->keyword[iter]);

		int dim = query->searchDim[iter];
		GpuIndexDimensionEntry indexDimEntry = indexDimensionEntry_vec[dim];
		int keyword = y_dim_value + dim * max_value_per_dimension;

		if(down_pos == 0 && up_pos == 0){
			int invert_list_start = (keyword == 0 ? 0 : invert_list_idx[keyword-1]);
			int invert_list_end = invert_list_idx[keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size + (invert_list_size % block_size != 0);



			for(int j = 0; j < process_round; j++){
				int idx = invert_list_start+j*block_size+tid;
				if(idx < invert_list_end){
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = bid * (*dev_maxFeatureID) + inv_ent;
					//atomicAdd(&(query_feature[target_idx].count),1);
					query_feature[target_idx].count += 1;

					//if(bid == 1) printf("query %d meet doc %d in origin\n",bid,inv_ent);


				}
			}
		}

		if(point_search)
		{
			__syncthreads();
			continue;
		}

		// going downward
		int down_compute = dev_numOfDocToExpand[0];
		int down_value = y_dim_value - down_pos; // move the position that last iteration possessed.

		//while( down_value - 1 >= dev_minDomainForAllDimension[dim] && // make sure dimension is above minimum dimension value
		while( down_value - 1 >= indexDimEntry.minDomain && // make sure dimension is above minimum dimension value
				down_compute > 0 &&	// make sure the number of compute element is above 0
				down_value > y_dim_value - bound_down_search )
		{
			down_value--;
			int down_keyword = down_value + dim * max_value_per_dimension;

			int invert_list_start = down_keyword == 0 ? 0 : invert_list_idx[down_keyword-1];
			int invert_list_end = invert_list_idx[down_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size + (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float true_dist = distance_func( y_dim_value, down_value, dim_dist_func, dim_lb_dist_func, dim_weight);

			for(int j = 0; j < process_round; j++){
				int idx = invert_list_start+j*block_size+tid;
				if(idx < invert_list_end){
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = bid * (*dev_maxFeatureID) + inv_ent;
					query_feature[target_idx].count += 1;
					query_feature[target_idx].ACD += true_dist;


				}
			}

			down_compute -= invert_list_size;
		}

		// going upward
		int up_compute = *dev_numOfDocToExpand;
		int up_value = y_dim_value + up_pos; // move the position that last iteartion possessed.

		while( up_value + 1 <= indexDimEntry.maxDomain && // make sure dimension is below maximum dimension value
				up_compute > 0 &&// make sure the number of compute element is above 0
				up_value < y_dim_value + bound_up_search )
		{
			up_value++;
			int up_keyword = up_value + dim * max_value_per_dimension;

			int invert_list_start = up_keyword == 0 ? 0 : invert_list_idx[up_keyword-1];
			int invert_list_end = invert_list_idx[up_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size + (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float true_dist = distance_func(y_dim_value, up_value, dim_dist_func, dim_ub_dist_func, dim_weight);

			for(int j = 0; j < process_round; j++){
				int idx = invert_list_start+j*block_size+tid;
				if(idx < invert_list_end){
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = bid * (*dev_maxFeatureID) + inv_ent;

					//atomicAdd(&(query_feature[target_idx].count),1);
					//atomicAdd(&(query_feature[target_idx].ACD),true_dist);
					query_feature[target_idx].count += 1;
					query_feature[target_idx].ACD += true_dist;

				}
			}

			up_compute -= invert_list_size;
		}

		__syncthreads();
		if(tid == 0){
			pos[iter].x =  y_dim_value - down_value;
			pos[iter].y = up_value - y_dim_value;
		}
	}
	__syncthreads();
	if ( tid < (*dev_dimensionNum_perQuery) )
	{
		query->lastPos[tid] = pos[tid];
	}
}
*/






//==============================following part: code recut, one kernel take one dimension
/**
 * NOTE : 1.now assume all query has the same number of dimensions, which is stored in *dev_totalDimension
 *
 *//*
__global__ void compute_mapping_saving_pos_KernelPerDim(
		QueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		int* dev_minDomainForAllDimension,
		int* dev_maxDomainForAllDimension) {

	//int bid = blockIdx.x; // mapped to different dimension of a query

	int tid = threadIdx.x;
	int block_size = blockDim.x;

	int qid  =  blockIdx.x/dev_dimensionNum_perQuery[0];
	int dimId= blockIdx.x%dev_dimensionNum_perQuery[0]; //mapped to different dimension index of A QUERY (note that: this dimId is to index the query dimension, i.e. by query->searchDim[dimId] to locate the true dimensions
	QueryInfo* query = query_list[qid];


	if (dimId < query->numOfDimensionToSearch) {
		//for(int iter = 0; iter < query->numOfDimensionToSearch; iter++){

		float dim_weight = query->dimWeight[dimId];
		float dim_dist_func = query->distanceFunc[dimId];
		float dim_lb_dist_func = query->lowerBoundDist[dimId];
		float dim_ub_dist_func = query->upperBoundDist[dimId];
		int bound_down_search = query->lowerBoundSearch[dimId];
		int bound_up_search = query->upperBoundSearch[dimId];
		int down_pos = query->lastPos[dimId].x;
		int up_pos = query->lastPos[dimId].y;
		int y_dim_value = rounding(query->keyword[dimId]);

		int dim = query->searchDim[dimId];

		//int keyword = y_dim_value + dim * MAX_DIM_VALUE;
		int keyword = y_dim_value + dim * max_value_per_dimension;


		if (down_pos == 0 && up_pos == 0) {
			int invert_list_start =
					keyword == 0 ? 0 : invert_list_idx[keyword - 1];
			int invert_list_end = invert_list_idx[keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureID[0]+inv_ent;
					//query_feature[target_idx].count += 1;
					atomicAdd(&(query_feature[target_idx].count), 1);

				}
			}
		}


		if (point_search) {
			//__syncthreads();
			//continue;
			return;
		}



		// going downward
		//int down_compute = invert_list_spec.numOfDocToExpand;
		int down_compute = (*dev_numOfDocToExpand);
		int down_value = y_dim_value - down_pos; // move the position that last iteartion possessed.

		while (down_value - 1 >= dev_minDomainForAllDimension[dim] // make sure dimension is above minimum dimension value
		&& down_compute > 0	// make sure the number of compute element is above 0
		&& down_value >= y_dim_value - bound_down_search) {
			down_value--;
			//int down_keyword = down_value + dim * MAX_DIM_VALUE;
			int down_keyword = down_value + dim * max_value_per_dimension;

			int invert_list_start =
					down_keyword == 0 ? 0 : invert_list_idx[down_keyword - 1];
			int invert_list_end = invert_list_idx[down_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float true_dist = distance_func(y_dim_value, down_value,
					dim_dist_func, dim_lb_dist_func, dim_weight);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureID[0]+inv_ent;
					//int target_idx = bid*invert_list_spec.maxFeatureID+inv_ent;
					//query_feature[target_idx].count += 1;
					//query_feature[target_idx].ACD += true_dist;
					atomicAdd(&(query_feature[target_idx].count), 1);
					atomicAdd(&(query_feature[target_idx].ACD), true_dist);
					//if(bid == 1) printf("query %d meet doc %d going down\n",bid,inv_ent);

				}
			}

			down_compute -= invert_list_size;
		}

		// going upward
		//int up_compute = invert_list_spec.numOfDocToExpand;
		int up_compute = (*dev_numOfDocToExpand);
		int up_value = y_dim_value + up_pos; // move the position that last iteartion possessed.

		//while(up_value + 1 <= invert_list_spec.maxDomainForAllDimension[dim] // make sure dimension is below maximum dimension value
		while (up_value + 1 <= dev_maxDomainForAllDimension[dim] // make sure dimension is below maximum dimension value
		&& up_compute > 0 // make sure the number of compute element is above 0
		&& up_value <= y_dim_value + bound_up_search) {
			up_value++;
			//int up_keyword = up_value + dim * MAX_DIM_VALUE;
			int up_keyword = up_value + dim * max_value_per_dimension;

			int invert_list_start =
					up_keyword == 0 ? 0 : invert_list_idx[up_keyword - 1];
			int invert_list_end = invert_list_idx[up_keyword];
			int invert_list_size = invert_list_end - invert_list_start;
			int process_round = invert_list_size / block_size
					+ (invert_list_size % block_size != 0);

			// calcuate the distance from the current point to the query along this dimension
			float true_dist = distance_func(y_dim_value, up_value,
					dim_dist_func, dim_ub_dist_func, dim_weight);

			for (int j = 0; j < process_round; j++) {
				int idx = invert_list_start + j * block_size + tid;
				if (idx < invert_list_end) {
					InvlistEnt inv_ent = invert_list[idx];
					int target_idx = qid*dev_maxFeatureID[0]+inv_ent;
					int target_idx = bid*invert_list_spec.maxFeatureID+inv_ent;
					 query_feature[target_idx].count += 1;
					 query_feature[target_idx].ACD += true_dist;
					atomicAdd(&(query_feature[target_idx].count), 1);
					atomicAdd(&(query_feature[target_idx].ACD), true_dist);
					//if(bid == 1) printf("query %d meet doc %d going up\n",bid,inv_ent);
				}
			}

			up_compute -= invert_list_size;
		}

		__syncthreads();
		if (tid == 0) {
			query->lastPos[dimId].x = y_dim_value - down_value;
			query->lastPos[dimId].y = up_value - y_dim_value;
			//printf("Dimension %d with dim value %d is up value: %d and down value %d\n",bid,y_dim_value,query->lastPos[dim].x,query->lastPos[dim].y);
		}
	}

}
*/


/*
 *
//this is specialized for a function
__device__ void blk_sort_shared_memory_V1(float* shared_array, float* temp_array, int size)
{
	int rounds = ceil(log2((double)size));

	int base = 1;
	for(int i = 1; i <= rounds; i++){

		if(threadIdx.x % (2*base) == 0){
			int start_idx_x = threadIdx.x;
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
					temp_array[output_ptr++] = shared_array[y_pointer++];
				else if (y_pointer >= end_idx_y)
					temp_array[output_ptr++] = shared_array[x_pointer++];
				else if (shared_array[x_pointer] < shared_array[y_pointer])
					temp_array[output_ptr++] = shared_array[y_pointer++];
				else
					temp_array[output_ptr++] = shared_array[x_pointer++];
			}
		}
		__syncthreads();
		if(threadIdx.x < size){
			shared_array[threadIdx.x] = temp_array[threadIdx.x];
		}
		__syncthreads();
		base *= 2;
	}
}

 *
 */

/*
 *
 *
**
 * author: zhou jignbo
 * //this is v2 version which is consistent with GPUManager::bi_direction_query_KernelPerQuery_V2() function
 *
 *
__global__ void output_result_bidrection_search_KernelPerQuery_V2(
	QueryFeatureEnt* query_feature,
	QueryInfo** query_set, // remeber the counting position
	int* perthread_ending_idx, //threshold_count
	int* perblock_ending_idx, //query_count
	Result* result_vec,//record the lower bound and upper bound
	int threshold,
	int* dev_minDomainForAllDimension,
	int* dev_maxDomainForAllDimension)
{

	// dynamic allocate shared memory. specify outside in kenel calls
	extern __shared__ float queryBoundMem[];
	float* queryLowerBound = queryBoundMem;
	float* queryUpperBound = &queryBoundMem[dev_dimensionNum_perQuery[0]];
	float* temp_shared = &queryBoundMem[2*dev_dimensionNum_perQuery[0]];


	QueryInfo* queryInfo = query_set[blockIdx.x];

	int block_num_search_dim = queryInfo->numOfDimensionToSearch;


	blk_get_eachQueryDimensionBound( queryInfo, block_num_search_dim,  dev_minDomainForAllDimension,
			dev_maxDomainForAllDimension, queryLowerBound,  queryUpperBound );

	// sort the lower and up bound
	// also calculate lower bound for this query, total distance for current retrieve point;
	float lower_bound_sum = blk_sort_and_sum_shared_memory_V2(queryLowerBound, temp_shared, *dev_dimensionNum_perQuery);
	blk_sort_shared_memory(queryUpperBound, temp_shared, *dev_dimensionNum_perQuery);

	//scan and compact non-zeor item into result_vec, this is not the final result
	blk_compact_resultVec_V2( query_feature, perthread_ending_idx, threshold, result_vec);
	blk_get_resultVec_V2(block_num_search_dim,perblock_ending_idx, lower_bound_sum, queryLowerBound, queryUpperBound, result_vec);

	//blk_compute_direct_resultVec_V2(query_feature,
	//				  perthread_ending_idx,
	//				  threshold,
	//				  block_num_search_dim,
	//				  lower_bound_sum,
	//				  queryLowerBound, queryUpperBound ,
	//				  result_vec
	//		);

}

 *
 */


/*
 *
 *
__device__ float blk_sum_shared_memory_V2( float* temp_array, int size){

	for (unsigned int s = size / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			temp_array[threadIdx.x] += temp_array[threadIdx.x + s];
		}
	}
	__syncthreads();
	return temp_array[0];
}

__device__ float inline blk_sort_and_sum_shared_memory_V2(float* shared_array, float* temp_array, int size){

	blk_sort_shared_memory(shared_array, temp_array, size);

	return blk_sum_shared_memory_V2( temp_array, size);
}




/*

*
 *scan and compact non-zeor item into result_vec, this is not the final result

__device__ void blk_compact_resultVec_V2(
		 QueryFeatureEnt* query_feature,
		 int* perthread_ending_idx, //threshold_count
		 int threshold,
		 Result* _result_vec//record the lower bound and upper bound
		 ) {

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int start_idx = pos == 0 ? 0 : perthread_ending_idx[pos - 1];

	int block_start = blockIdx.x * (*dev_maxFeatureID);
	int block_end = block_start + (*dev_maxFeatureID);
	int round = (*dev_maxFeatureID) / blockDim.x
			+ ((*dev_maxFeatureID) % blockDim.x != 0);

	for (int i = 0; i < round; i++) {
		int idx = block_start + i * blockDim.x + threadIdx.x;

		if (idx < block_end) {
			int count = query_feature[idx].count;

			if (count > threshold) {
				Result new_result;
				new_result.query = blockIdx.x;
				new_result.feature_id = i * blockDim.x + threadIdx.x;
				new_result.count = count;
				new_result.lb = query_feature[idx].ACD;		//temporary store

				_result_vec[start_idx] = new_result;
				start_idx++;

			}

		}

	}
	__syncthreads();

}
*/
/*
__device__ void inline blk_compute_direct_resultVec_V2(
		QueryFeatureEnt* query_feature,
				 int* perthread_ending_idx, //threshold_count
				 int threshold,
				 int block_num_search_dim,
				 float lower_bound_sum,
				 float* queryLowerBound,float* queryUpperBound ,
				 Result* result_vec//record the lower bound and upper bound
		){

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
		int start_idx = pos == 0 ? 0 : perthread_ending_idx[pos - 1];

		int block_start = blockIdx.x * (*dev_maxFeatureID);
		int block_end = block_start + (*dev_maxFeatureID);
		int round = (*dev_maxFeatureID) / blockDim.x
				+ ((*dev_maxFeatureID) % blockDim.x != 0);

	for(int i = 0; i < round; i++)
	{
		int idx = block_start + i * blockDim.x + threadIdx.x;
		if(idx < block_end)
		{
			int count = query_feature[idx].count;

			if(count > threshold)
			{
				Result new_result;
				new_result.query = blockIdx.x;
				new_result.feature_id = i * blockDim.x + threadIdx.x;
				new_result.count = count;

				float ACD = query_feature[idx].ACD;

				// DL(f) = DL(us) - sum(max count number of dt) + ACD(f)
				float lb = lower_bound_sum;
				for(int m = 0; m < count; m++)
				{

					lb -= queryLowerBound[m];

				}
				lb += ACD;

				// DU(f) = ACD(f) + sum(max 128 - c of upper bound)
				float ub = ACD;
				for(int m = 0; m < block_num_search_dim - count; m++)
				{
					ub += queryUpperBound[m];
				}

				new_result.lb = lb;
				new_result.ub = ub;

				result_vec[start_idx] = new_result;
				start_idx++;


				//comment by jingbo
				//printf("new_result.lb: %5.1f new_result.ub:%5.1f \n", new_result.lb, new_result.ub);
			}

		}
	}


}*/
/*

__device__ void blk_get_resultVec_V2(int block_num_search_dim, int* perblock_ending_idx,
		float lower_bound_sum, float* queryLowerBound,float* queryUpperBound ,
		Result* result_vec){

	int block_start = (blockIdx.x == 0 ? 0 : perblock_ending_idx[blockIdx.x - 1]);
	int block_end = perblock_ending_idx[blockIdx.x];
	int interval = block_end - block_start;

	int round = (interval) / blockDim.x
				+ ((interval) % blockDim.x != 0);


	for (int i = 0; i < round; i++) {
		int idx = block_start + i * blockDim.x + threadIdx.x;
		if (idx < block_end) {
			Result new_result = result_vec[idx];

			float ACD =new_result.lb;
			int count = new_result.count;
			// DL(f) = DL(us) - sum(max count number of dt) + ACD(f)
			float lb = lower_bound_sum;
			for (int m = 0; m < count; m++) {

				lb -= queryLowerBound[m];
			}
			lb += ACD;

			// DU(f) = ACD(f) + sum(max 128 - c of upper bound)
			float ub = ACD;
			for (int m = 0; m < block_num_search_dim - count; m++) {
				ub += queryUpperBound[m];
			}

			new_result.lb = lb;
			new_result.ub = ub;
			result_vec[idx]=new_result;
		}
	}


}
*/

/**
 * part code:

	if(threadIdx.x < block_num_search_dim)
	{

		int query_dim_value = queryInfo->keyword[threadIdx.x];
		float dim_dist_func = queryInfo->distanceFunc[threadIdx.x];
		float dim_weight = queryInfo->dimWeight[threadIdx.x];
		float min_bound_dist_func = queryInfo->lowerBoundDist[threadIdx.x];
		float max_bound_dist_func = queryInfo->upperBoundDist[threadIdx.x];
		int dim_upperBoundSearch = queryInfo->upperBoundSearch[threadIdx.x];// search bound when going up, add by jingbo
		int dim_lowerBoundSearch = queryInfo->lowerBoundSearch[threadIdx.x];// search boudn when going down, add by jingbo
		//float min_bound_dist_func;
		//float max_bound_dist_func;
		int dim = queryInfo->searchDim[threadIdx.x];

		queryLowerBound[threadIdx.x] = 0.;
		queryUpperBound[threadIdx.x] = 0.;
		temp_shared[threadIdx.x] = 0.;

		int2 q_pos = queryInfo->lastPos[threadIdx.x];
		// make sure the bound is correct when upward and downward search all reach the maximum, comment by jingbo
		// modified to the min compare to modified.cu which use max
		bool isReachMin = (q_pos.x >=  dim_lowerBoundSearch);//(q_pos.x	== (query_dim_value - dev_minDomainForAllDimension[dim]));
		bool isReachMax = (q_pos.y >= dim_upperBoundSearch);//(q_pos.y == (dev_maxDomainForAllDimension[dim] - query_dim_value));

		float min_lb = (float) INT_MAX;

		if (isReachMin && isReachMax) {	min_lb = max(q_pos.x, q_pos.y);	}
		else if (isReachMin) {	min_lb = q_pos.y;}
		else if (isReachMax) { min_lb = q_pos.x;}
		else {	min_lb = min(q_pos.x, q_pos.y);	}


		int min_value = 0;
		queryLowerBound[threadIdx.x] = distance_func( min_lb, min_value, dim_dist_func, min_bound_dist_func, dim_weight );


		int max_ub = max( (query_dim_value - dev_minDomainForAllDimension[dim]), (dev_maxDomainForAllDimension[dim] - query_dim_value) );
		queryUpperBound[threadIdx.x] =distance_func ( max_ub, min_value, dim_dist_func, max_bound_dist_func, dim_weight );

	}

*/
