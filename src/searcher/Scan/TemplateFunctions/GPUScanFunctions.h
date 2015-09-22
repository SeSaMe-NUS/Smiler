/*
 * GPUFunctions.h
 *
 *  Created on: Jun 14, 2014
 *      Author: zhoujingbo
*/

#ifndef GPUFUNCTIONS_H_
#define GPUFUNCTIONS_H_
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include "../../lib/bucket_topk/bucket_topk.h"
#include "../UtlScan.h"
#include "../DistFunc.h"

#include <vector>
using namespace std;



template<class T, class DISTFUNC>
__global__ void computeScanDist(
	const T* queryData, const int* queryData_endIdx, const int* query_blade_data_id,
	const T* blade_data, const int* data_endIdx,
	DISTFUNC distFunc,
	float* result_dist_vec,//output
	int* result_vec_endIdx
 	 );


void inline getMinMax(device_vector<float>& d_data_vec, float & min, float& max);

//
//template <class T, class DISTFUNC>
//void GPU_computeTopk(vector<vector<T> >& query_vec, vector<int>& query_blade_id_vec,
//		device_vector<T> d_blade_data_vec,  device_vector<int>& d_blade_data_vec_endIdx, vector<int>& d_blade_data_vec_size,
//		 vector<int>& topk_vec ,
//		 DISTFUNC distFunc,
//		 vector<vector<topNode> >& _topk_results//output
//		);

template <class T, class DISTFUNC>
 void GPU_computeTopk(vector<vector<T> >& query, vector<int>& query_blade_id_vec,
 		 vector<vector<T> >& bladeData, vector<int>& topk,
 		DISTFUNC distFunc,
 		 vector<vector<topNode> >& _topk_result_idx);



#include "GPUScanFunctions.inc"

#endif  /* GPUFUNCTIONS_H_ */
