/*
 * GPUScanFunctions.cpp
 *
 *  Created on: Jun 14, 2014
 *      Author: zhoujingbo
 */

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
using namespace std;

#include <cuda_profiler_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>

using namespace thrust;
//#define THREAD_PER_BLK 256  // must be greater or equal to MAX_DIM and not exceed 1024
#include "../../GPUMacroDefine.h"

#define BUCKTOPK_SPLIT_THRESHOLD 512000000


/*
GPUScanFunctions::GPUScanFunctions() {
	// TODO Auto-generated constructor stub

}

GPUScanFunctions::~GPUScanFunctions() {
	// TODO Auto-generated destructor stub
}*/





/*
 * shared memory size: the maximum dimension of one query
*/
 template<class T, class DISTFUNC>
__global__ void computeScanDist(
	const T* query_vec, const int* query_vec_endIdx, const int* query_blade_data_id,
	const T* blade_data_vec, const int* blade_data_vec_endIdx,
	DISTFUNC distFunc,
	float* _distanceComp_vec,//out put
	int* disttanceComp_vec_endIdx//input, indicate the endIdx of _distanceComp_vec for each query
 	 ){
	 int bid = blockIdx.x;
	 extern __shared__ T sm_query[];
	 int query_start_id = (bid == 0)? 0 : query_vec_endIdx[bid - 1];
	 int query_end_id = query_vec_endIdx[blockIdx.x];

	 int queryLen = query_end_id-query_start_id;

	 int round = queryLen/blockDim.x + (queryLen%blockDim.x!=0);

	 for(int i=0;i<round;i++){
		 int idx = i*blockDim.x + threadIdx.x;

		 if(idx<queryLen){
			 sm_query[idx] = query_vec[idx+query_start_id];
		 }

	 }
	 __syncthreads();

	 int blade_id = query_blade_data_id[blockIdx.x];

	 int blade_start_idx = (blade_id == 0)? 0:blade_data_vec_endIdx[blade_id-1];
	 int blade_end_idx = blade_data_vec_endIdx[blade_id];
	 int blade_len = blade_end_idx - blade_start_idx;

	 int result_start_idx = (blockIdx.x == 0)? 0:disttanceComp_vec_endIdx[blockIdx.x - 1];

	 int query_round = (blade_len)/blockDim.x + ((blade_len)%blockDim.x!=0);

	 for(int i=0;i<query_round;i++){
		 int idx = i*blockDim.x + threadIdx.x;
			//left MAX_PRE_MAX_STEP out of GPU to avoid exceeding the bounding when query the reference time series
		 if(idx<blade_len-queryLen-MAX_PRE_MUL_STEP+1){


			 float res = distFunc.dist(sm_query, 0, blade_data_vec, blade_start_idx+idx, queryLen);


			 _distanceComp_vec[result_start_idx + idx] = res;

		 }
	 }

 }





template <class T>
struct ValueOfSeletion{
__host__ __device__ T valueOf(T data)
{
   		return data;
	}
};



void inline getMinMax(device_vector<float>& d_data_vec, float & min, float& max) {

 	device_vector<float>::iterator minData =
 			thrust::min_element(d_data_vec.begin(), d_data_vec.end());

 	device_vector<float>::iterator maxData = thrust::max_element(d_data_vec.begin(), d_data_vec.end());
 	device_vector<float> d_min(1), d_max(1);
 	thrust::copy(minData, minData + 1, d_min.begin());
 	thrust::copy(maxData, maxData + 1, d_max.begin());
 	host_vector<float> h_min,h_max;
 	h_min = d_min, h_max = d_max;

 	min = h_min[0];
 	max = h_max[0];

 	if(min<1.0e-3&&min>0){
 		min = 0;
 	}
 }

template<typename T>
void inline splitDeviceVector(device_vector<T>& master, device_vector<T>& first, device_vector<T>& second){

	first.resize(master.size()/2);
	second.resize(master.size()/2+master.size()%2);

	thrust::copy(master.begin(),master.begin()+master.size()/2,first.begin());
	thrust::copy(master.begin()+master.size()/2,master.end(),second.begin());

}


void print_d_distComp_vec(device_vector<float>& d_distComp_vec,device_vector<int>& d_distComp_vec_startIdx){
	cout << "with debug purpose print_d_distComp_vec()" << endl;

	 host_vector<float> h_d = d_distComp_vec;
	 host_vector<int> h_idx = d_distComp_vec_startIdx;
	 cout<<"d_distComp_vec.size():"<<d_distComp_vec.size()<<endl;
	 cout<<"d_distComp_vec_startIdx[0]:"<<h_idx[0]<<endl;
	 cout<<"d_distComp_vec_startIdx[1]:"<<h_idx[1]<<endl;
	 cout<<"d_distComp_vec_startIdx[2]:"<<h_idx[2]<<endl;

	for (int i = 0; i < d_distComp_vec.size(); i++) {
			if((i>63324&&i<63380)||(i>9320&&i<9340)){
				printf("dist_Comp_vec[%d]=%f \n",i,h_d[i]);
			}
		}
}


template <class T, class DISTFUNC>
void GPU_computeTopk(vector<vector<T> >& query_vec, vector<int>& query_blade_id_vec,
		device_vector<T>& d_blade_data_vec,  device_vector<int>& d_blade_data_vec_endIdx, vector<int>& d_blade_data_vec_size,
		 vector<int>& topk_vec ,
		 DISTFUNC distFunc,
		 vector<vector<topNode> >& _topk_results//output
		){

	//preparing query for for GPU
	vector<T> std_query_vec;
  	vector<int> std_query_vec_idx(query_vec.size());
  	int maxQueryLen = 0;


  	for(int i=0;i<query_vec.size();i++){
  		for(int j=0;j<query_vec[i].size();j++){

  			std_query_vec.push_back(query_vec[i][j]);
  		}

  		if(maxQueryLen<query_vec[i].size()){maxQueryLen = query_vec[i].size();}

  		std_query_vec_idx[i] = (query_vec[i].size());
  	}

  	device_vector<T> d_query_vec(std_query_vec);
  	device_vector<int> d_query_vec_endIdx (std_query_vec_idx);
  	device_vector<int> d_query_blade_id (query_blade_id_vec);
  	thrust::inclusive_scan(d_query_vec_endIdx.begin(),d_query_vec_endIdx.end(),d_query_vec_endIdx.begin());

  	//prepare buffer for storing computed distance
  	host_vector<int> h_distComp_vec_size(query_vec.size());
  	for(int i=0;i<query_vec.size();i++){
  		int queryItem_blade_id = query_blade_id_vec[i];
  		//left MAX_PRE_MAX_STEP+query_vec[i].size() out of GPU to avoid exceeding the bounding when query the reference time series
  		h_distComp_vec_size[i] = d_blade_data_vec_size[queryItem_blade_id]-query_vec[i].size()-MAX_PRE_MUL_STEP+1;
  	}

  	device_vector<int> d_distComp_vec_size = h_distComp_vec_size;
	device_vector<int> d_distComp_vec_startIdx(h_distComp_vec_size.size());
	device_vector<int> d_distComp_vec_endIdx(h_distComp_vec_size.size());

  	thrust::exclusive_scan(d_distComp_vec_size.begin(),d_distComp_vec_size.end(),d_distComp_vec_startIdx.begin());
  	thrust::inclusive_scan(d_distComp_vec_size.begin(),d_distComp_vec_size.end(),d_distComp_vec_endIdx.begin());

  	host_vector<int> h_distComp_vec_startIdx(h_distComp_vec_size.size());
  	h_distComp_vec_startIdx = d_distComp_vec_startIdx;

  	device_vector<float> d_distComp_vec(d_distComp_vec_endIdx[d_distComp_vec_endIdx.size()-1]);


  	//kernel function
  	computeScanDist<<<query_vec.size(),THREAD_PER_BLK, maxQueryLen*sizeof(T)>>>(
  		raw_pointer_cast(d_query_vec.data()),
  		raw_pointer_cast(d_query_vec_endIdx.data()),
  		raw_pointer_cast(d_query_blade_id.data()),
  		raw_pointer_cast(d_blade_data_vec.data()),
  		raw_pointer_cast(d_blade_data_vec_endIdx.data()),
  		distFunc,
  		raw_pointer_cast(d_distComp_vec.data()),
  		raw_pointer_cast(d_distComp_vec_endIdx.data())
  	 	);

  	//print_d_distComp_vec(d_distComp_vec,d_distComp_vec_startIdx);//with debug purpose

  	float min,max;
  	getMinMax(d_distComp_vec, min, max);


  	device_vector<float> min_vec(query_vec.size(),min);
  	device_vector<float> max_vec(query_vec.size(),max);

  	device_vector<int> d_topk_vec(topk_vec);
  	device_vector<int> d_topk_distComp_idSet;
  	int topkSum = thrust::reduce(d_topk_vec.begin(),d_topk_vec.end());
  	device_vector<float> d_topk_distComp_vec(topkSum);



  	//cout << "d_distComp_vec size: " << d_distComp_vec.size() << " memory:"
	//		<< d_distComp_vec.size() * sizeof(d_distComp_vec[0]) << endl;

	if (d_distComp_vec.size() < BUCKTOPK_SPLIT_THRESHOLD) {
		bucket_topk(&d_distComp_vec, ValueOfSeletion<float>(), &min_vec,
				&max_vec, &d_topk_vec, &d_distComp_vec_startIdx,
				&d_distComp_vec_endIdx, query_vec.size(),
				&d_topk_distComp_idSet);



	} else {

		device_vector<int> d_topk_vec_first;
		device_vector<int> d_topk_vec_second;
		splitDeviceVector(d_topk_vec, d_topk_vec_first, d_topk_vec_second);

		device_vector<int> d_distComp_vec_startIdx_first;
		device_vector<int> d_distComp_vec_startIdx_second;
		splitDeviceVector(d_distComp_vec_startIdx,
				d_distComp_vec_startIdx_first, d_distComp_vec_startIdx_second);

		device_vector<int> d_distComp_vec_endIdx_first;
		device_vector<int> d_distComp_vec_endIdx_second;
		splitDeviceVector(d_distComp_vec_endIdx, d_distComp_vec_endIdx_first,
				d_distComp_vec_endIdx_second);

		device_vector<float> d_min_first, d_min_second;
		splitDeviceVector(min_vec, d_min_first, d_min_second);

		device_vector<float> d_max_first, d_max_second;
		splitDeviceVector(max_vec, d_max_first, d_max_second);

		device_vector<int> d_topk_distComp_idSet_first;
		device_vector<int> d_topk_distComp_idSet_second;
		bucket_topk(&d_distComp_vec, ValueOfSeletion<float>(),
				&d_min_first, &d_max_first,
				&d_topk_vec_first, &d_distComp_vec_startIdx_first,
				&d_distComp_vec_endIdx_first, d_topk_vec_first.size(),
				&d_topk_distComp_idSet_first);

		bucket_topk(&d_distComp_vec, ValueOfSeletion<float>(),
				&d_min_second, &d_max_second,
				&d_topk_vec_second, &d_distComp_vec_startIdx_second,
				&d_distComp_vec_endIdx_second, d_topk_vec_second.size(),
				&d_topk_distComp_idSet_second);

		d_topk_distComp_idSet.resize(
				d_topk_distComp_idSet_first.size()
						+ d_topk_distComp_idSet_second.size());
		thrust::copy(d_topk_distComp_idSet_first.begin(),
				d_topk_distComp_idSet_first.end(),
				d_topk_distComp_idSet.begin());
		thrust::copy(d_topk_distComp_idSet_second.begin(),
				d_topk_distComp_idSet_second.end(),
				d_topk_distComp_idSet.begin()
						+ d_topk_distComp_idSet_first.size());
	}



	thrust::copy(thrust::make_permutation_iterator(d_distComp_vec.begin(),d_topk_distComp_idSet.begin()),
				thrust::make_permutation_iterator(d_distComp_vec.begin(),d_topk_distComp_idSet.end()),
				d_topk_distComp_vec.begin()
		);


     host_vector<int> h_topk_distComp_idSet = d_topk_distComp_idSet;
     host_vector<float> h_topk_distComp_vec = d_topk_distComp_vec;


     int res_count_idx = 0;
     _topk_results.clear();
     _topk_results.resize(topk_vec.size());
     for(int i=0;i<topk_vec.size();i++){
     	for(int j=0;j<topk_vec[i];j++){
     		int id = h_topk_distComp_idSet[res_count_idx] -  h_distComp_vec_startIdx[i];
     		float dis = h_topk_distComp_vec[res_count_idx];

     		topNode tn(dis,id);
     		_topk_results[i].push_back(tn);
     		res_count_idx++;
     	}
       std::sort(_topk_results[i].begin(),_topk_results[i].end(),CompareTopNode());
     }

}





/**
 * TODO:
 *
 * @parameters:
 * query_vec: a set of queries
 * query_blade_id_vec: the query is corresponding to which data blade
 * bladeData_vec: several data blades, the mapping from query to corresponding blades is recorded in query_blade_id_vec
 */
  template <class T, class DISTFUNC>
  void GPU_computeTopk(vector<vector<T> >& query_vec, vector<int>& query_blade_id_vec,
 		 vector<vector<T> >& bladeData_vec,
 		 vector<int>& topk_vec ,
 		 DISTFUNC distFunc,
 		 vector<vector<topNode> >& _topk_results//output
 		 ){


	//preparing blade data for GPU
  	vector<T> std_blade_data_vec;
  	vector<int> std_blade_data_size;
  	for(int i=0;i<bladeData_vec.size();i++){
  		for(int j=0;j<bladeData_vec[i].size();j++){
  			std_blade_data_vec.push_back(bladeData_vec[i][j]);
  		}
  		std_blade_data_size.push_back(bladeData_vec[i].size());
  	}

  	device_vector<T> d_blade_data_vec(std_blade_data_vec);
  	device_vector<int> d_blade_data_vec_size(std_blade_data_size);;
  	device_vector<int> d_blade_data_vec_endIdx(std_blade_data_size.size());
  	thrust::inclusive_scan(d_blade_data_vec_size.begin(),d_blade_data_vec_size.end(),d_blade_data_vec_endIdx.begin());


  	GPU_computeTopk( query_vec,  query_blade_id_vec,
  			 d_blade_data_vec,  d_blade_data_vec_endIdx, std_blade_data_size,
  			 topk_vec ,
  			 distFunc,
  			 _topk_results//output
  			);

  }


