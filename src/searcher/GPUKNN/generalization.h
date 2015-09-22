#ifndef GENERALIZATION_H
#define GENERALIZATION_H

#include <stdio.h>
#include <vector>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include "UtlGPU.h"

#include <cuda.h>
#include <device_functions.h>

using namespace std;
using namespace thrust;

#include "../GPUMacroDefine.h"
//#define THREAD_PER_BLK 256  // must be greater or equal to MAX_DIM and not exceed 1024


/**
 * print error text
 */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



/* the entry of a inverted list, this number must not exceed QuerySpecGPU.maxFeatureID */
typedef int InvlistEnt;

/* the entry of mapping from query to feature */
struct QueryFeatureEnt
{
	int count; // under current setup, count is actually integer and max is MAX_DIM
	float ACD; // accumulated distance

	__host__ __device__ QueryFeatureEnt(){
		count = 0;
		ACD = 0;
	}

	__host__ __device__ void print(){
		printf("count %d ACD %f\n",count, ACD);
	}
};

//when the inverted list is built on unit value, i.e. one unit value is one keyword, use this struct
struct DataToIndex_keywordMap_bucketUnit{

	__host__ __device__ inline int mapping(float x, float bucketWidth){
		return (int)(x+0.5);
	}
};

//when the inverted list is built on unit value, i.e. one unit value is one keyword, use this struct
struct IndexToData_lastPosMap_bucketUnit{

	__host__ __device__ inline float map_indexToData_up(int index_up,float bucketWidth){
		return (float)index_up;
	}
	__host__ __device__ inline float map_indexToData_down(int index_down, float bucketWidth){
		return (float)index_down;
	}
};

//when the inverted index is built by bucket with width, use this struct
struct DataToIndex_keywordMap_bucket{

	__host__ __device__ inline int mapping(float x, float bucketWidth){
		return (int)((x)/bucketWidth);
	}
};

//when the inverted index is built by bucket with width, use this struct to update ACD value
//compute the value, but exclusive the index part bucket
struct IndexToData_lastPosMap_bucket_exclusive{

	__host__ __device__ inline float map_indexToData_up(int index_up,float bucketWidth){
		return (float)index_up*bucketWidth;
	}
	__host__ __device__ inline float map_indexToData_down(int index_down, float bucketWidth){
		return (float)(index_down+1)*bucketWidth;
	}
};

//when the inverted index is built by bucket with width, use this struct to compute lower and uppper bound
//compute the value, but inclusive the index part bucket
struct IndexToData_lastPosMap_bucket_inclusive{

	__host__ __device__ inline float map_indexToData_up(int index_up,float bucketWidth){
		return (float)(index_up+1)*bucketWidth;
	}
	__host__ __device__ inline float map_indexToData_down(int index_down, float bucketWidth){
		return (float)(index_down)*bucketWidth;
	}
};


struct Lp_distance {

	__host__ __device__ Lp_distance() {

	}

// distance function for values on the same dimension
	__host__ __device__ float inline dist(float a, float b, float func_type, float bound, float dim_weight) {
		{
			float result;
			float diff = fabs(a - b);

			if (func_type == 0) {
				result = (diff <= bound) ? 0 : 1; // the bound step function
				return result;
			} else if (func_type == 1) {

				result = (diff <= bound) ? 0 : (diff - bound); // the bound step function

			} else if (func_type == 2) {

				result = (diff <= bound) ? 0 : (diff - bound) * (diff - bound);

			} else {
				result = (diff <= bound) ? 0 : powf(diff - bound, func_type);
			}

			//result = (diff < bound) ? 0 : 1; // the bound step function, how to involve the bound
			result = (diff == 0) ? 0 : result; // make sure the 0 case is correct
			result *= dim_weight; // add dimension weight

			return result;
		}
	}

};

//===struct for aggregation function in blk_aggregation_sharedMemory()
struct Blk_Sum{
	__device__ float op(float a, float b){
		return (a+b);
	}
};

struct Blk_Max{
	__device__ float op(float a, float b){
		return fmaxf(a,b);
	}
};



void initialize_dev_constMem( const InvertListSpecGPU& spec );
void init_dev_dimNumPerQuery_constMem(int dimensionNum_perGroup);

void freeGPUMemory();



__global__ void printQueryInfo(WindowQueryInfo** query_list, int size);
__global__ void printConstMem();





__device__ void blk_sort_inSharedMemory(float* data_array, float* temp_array, int size);
__device__ float blk_sum_sharedMemory( float* data_array, int size);
template <class AGG>
__device__ float  blk_aggregation_sharedMemory( float* data_array, int size, AGG agg);


__device__ bool checkQuery( WindowQueryInfo** query_list );




template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__global__ void compute_mapping_saving_pos_KernelPerDim_template(
		WindowQueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc);



template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
__device__ void blk_get_eachQueryDimensionBound_template(WindowQueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound, float* _queryUpperBound,
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc
		);

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
	);



__global__ void prefix_count_KernelPerQuery(
		QueryFeatureEnt* query_feature,
				int* count_vec,
				int* blk_count_vec,
				int threshold
				);


__global__ void output_result_point_search(
		QueryFeatureEnt* query_feature,
		int* ending_idx,
		Result* result_vec,
		int threshold);





__global__ void depressed_terminateCheck_afterSort_KernelPerQuery(Result* ub_sorted,Result* lb_sorted, int* end_idx, int* output, int K, int round_num);
__global__ void terminateCheck_kSelection_KernelPerQuery(
		Result* boundData,
		Result* temp,//d_result_lb_sorted array
		int* end_idx,//query_result_count.data()
		int* output,//d_valid_query.data()
		int K);

__global__ void extract_topK_KernelPerQuery ( Result* ub_sorted, int* end_idx, Result* output, int K);



//=========following code: depressed functions====================

//this function should not be used
__global__ void depressed_prefix_count_KernelPerQuery_V1(
		QueryFeatureEnt* query_feature,
		int* count_vec,
		int* blk_count_vec,
		int threshold,
		GpuIndexDimensionEntry* indexDimensionEntry_vec

		);


//this function should not be used, but we keep this functin for experiment purpose
__global__ void depressed_compute_mapping_saving_pos_KernelPerDim(
		WindowQueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec
		);

__device__ inline int depressed_rounding(float x) {return (int)(x+0.5);}


// distance function for values on the same dimension
__device__ float inline depressed_distance_func(float a, float b, float func_type, float bound, float dim_weight){
	{
		float result;
		float diff = fabs(a - b);


		if(func_type==0){
			result =  (diff <= bound) ? 0 : 1; // the bound step function
			return result;
		}else if(func_type == 1){

			result =  (diff <= bound) ? 0 : (diff-bound); // the bound step function

		}else if(func_type == 2){

			result = (diff<=bound)?0:(diff-bound)*(diff-bound);

		}else {
			result = (diff<=bound)?0:powf(diff-bound, func_type);
		}


		//result = (diff < bound) ? 0 : 1; // the bound step function, how to involve the bound
		result = (diff == 0) ? 0 : result; // make sure the 0 case is correct
		result *= dim_weight; // add dimension weight

		return result;
	}
}


#endif





//junk code
//====2014.06.10
/*
__global__ void compute_mapping_saving_pos_KernelPerDim(
		QueryInfo** query_list,
		InvlistEnt* invert_list, int* invert_list_idx,
		QueryFeatureEnt* query_feature, bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec

		);
*/

/*
__device__ void blk_get_eachQueryDimensionBound(QueryInfo* queryInfo,int block_num_search_dim,
		GpuIndexDimensionEntry* indexDimensionEntry_vec,
		float* _queryLowerBound, float* _queryUpperBound );
*/


/*

__global__ void output_result_bidrection_search_KernelPerQuery(
	QueryFeatureEnt* query_feature,
	QueryInfo** query_set,
	int* ending_idx,
	Result* result_vec,
	int threshold,
	GpuIndexDimensionEntry* indexDimensionEntry_vec
	);
*/

/*

__global__ void compute_mapping_saving_pos_KernelPerQuery(
		QueryInfo** query_list,
		InvlistEnt* invert_list,
		int* invert_list_idx,
		QueryFeatureEnt* query_feature,
		bool point_search,
		int max_value_per_dimension,
		GpuIndexDimensionEntry* indexDimensionEntry_vec
		);

*/



//===========================================================

/*

__global__ void compute_mapping_saving_pos_KernelPerDim(
		QueryInfo** query_list,
		InvlistEnt* invert_list,
		int* invert_list_idx,
		QueryFeatureEnt* query_feature,
		bool point_search,
		int max_value_per_dimension,
		int* dev_minDomainForAllDimension,
		int* dev_maxDomainForAllDimension );
*/


/*

__global__ void prefix_count_KernelPerDim(QueryFeatureEnt* query_feature,int* count_vec, int threshold);

__global__ void compute_bound_bidrection_search_KernelPerDim(QueryInfo** query_set,float* queryLowerBound, float* queryUpperBound,int* dev_minDomainForAllDimension,
		int* dev_maxDomainForAllDimension);

// this function is used for bi directional search query
__global__ void output_result_bidrection_search_KernelPerDim(
	QueryFeatureEnt* query_feature,
	QueryInfo** query_set, // remeber the counting position
	int* ending_idx,Result* result_vec,
	int threshold,
	float* queryLowerBound, float* queryUpperBound);


__global__ void terminate_check_KernelPerDim(Result* ub_sorted,Result* lb_sorted, int* end_idx, int* output, int K, int round_num);


__global__ void extract_topK_KernelPerDim(Result* ub_sorted,int* end_idx,Result* output, int K);

__global__ void output_result_bidrection_search_KernelPerQuery_V2(
	QueryFeatureEnt* query_feature,
	QueryInfo** query_set, // remeber the counting position
	int* perthread_ending_idx, //threshold_count
	int* perblock_ending_idx, //query_count
	Result* result_vec,//record the lower bound and upper bound
	int threshold,
	int* dev_minDomainForAllDimension,
	int* dev_maxDomainForAllDimension);
*/

