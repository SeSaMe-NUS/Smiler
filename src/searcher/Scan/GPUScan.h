/*
 * GPUScan.h
 *
 *  Created on: Jun 16, 2014
 *      Author: zhoujingbo
*/

#ifndef GPUSCAN_H_
#define GPUSCAN_H_


#include <vector>
using namespace std;
//
//#include "TemplateFunctions/GPUScanFunctions.h"
//#include "DistFunc.h"
#include "UtlScan.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


//template <class T>
class GPUScan {
public:
	GPUScan();
	virtual ~GPUScan();



	//// void computTopk_eu(vector<vector<T> >& query, int k,
	// 		vector<T> & data);

//	 void computTopk_eu_int(vector<vector<int> >& query, int k,
//			vector<int> & data);

	 void computTopk_eu_float(vector<vector<float> >& query, int k,
	 		vector<float> & data) ;


//	 void computTopk_dtw_scBand_int(vector<vector<int> >& query, int k,
//	 		vector<int> & data, int sc_band);

	 void computTopk_dtw_scBand_float(vector<vector<float> >& query, int k,
	 		vector<float> & data, int sc_band);



	 // template <class T>
	 //dtw computation without scband constraints
//	void computTopk_dtw_modulus(vector<vector<T> >& query_vec,
//				vector<int>& query_blade_id_vec, vector<vector<T> >& bladeData_vec,
//				vector<int>& topk_vec,
//				vector<vector<topNode> >& _topk_resutls);

	void computTopk_dtw_modulus_float(vector<vector<float> >& query_vec,
					vector<int>& query_blade_id_vec, vector<vector<float> >& bladeData_vec,
					vector<int>& topk_vec,
					vector<vector<topNode> >& _topk_resutls);

//	void computTopk_dtw_modulus_int(vector<vector<int> >& query_vec,
//					vector<int>& query_blade_id_vec, vector<vector<int> >& bladeData_vec,
//					vector<int>& topk_vec,
//					vector<vector<topNode> >& _topk_resutls);


	 // template <class T>
	 //modulus computation with scband constraints, but do not compress the warping matrix
//	void computTopk_dtw_scBand_fullWM(vector<vector<T> >& query_vec,
//			vector<int>& query_blade_id_vec, vector<vector<T> >& bladeData_vec,
//			vector<int>& topk_vec, int sc_band,
//			vector<vector<topNode> >& _topk_resutls);

//	void computTopk_dtw_scBand_fullWM_int(vector<vector<int> >& query_vec,
//			vector<int>& query_blade_id_vec, vector<vector<int> >& bladeData_vec,
//			vector<int>& topk_vec, int sc_band,
//			vector<vector<topNode> >& _topk_resutls) ;

	void computTopk_dtw_scBand_fullWM_float(vector<vector<float> >& query_vec,
			vector<int>& query_blade_id_vec, vector<vector<float> >& bladeData_vec,
			vector<int>& topk_vec, int sc_band,
			vector<vector<topNode> >& _topk_resutls);


	// template <class T>
//	void computTopk_dtw_scBand(vector<vector<T> >& query_vec,
//			vector<int>& query_blade_id_vec, vector<vector<T> >& bladeData_vec,
//			vector<int>& topk_vec, int sc_band,
//			vector<vector<topNode> >& _topk_resutls);



//	void computTopk_dtw_scBand_int(vector<vector<int> >& query_vec,
//			vector<int>& query_blade_id_vec,
//			vector<vector<int> >& bladeData_vec,
//			vector<int>& topk_vec,
//			int sc_band,
//			vector<vector<topNode> >& _topk_result) ;


	void computTopk_dtw_scBand_float(vector<vector<float> >& query_vec,
			vector<int>& query_blade_id_vec,
			vector<vector<float> >& bladeData_vec,
			vector<int>& topk_vec,
			int sc_band,
			vector<vector<topNode> >& _topk_result);




	 //template <class T>
//	 void computeTopk_dtw_scBand(vector<vector<T> >& query_vec,vector<int>& query_blade_id_vec,
//			 thrust::device_vector<T>& d_blade_data_vec,  thrust::device_vector<int>& d_blade_data_vec_endIdx, vector<int>& d_blade_data_vec_size,
//	 		vector<int>& topk_vec, int sc_band,vector<vector<topNode> >& _topk_results);


//	 void computeTopk_dtw_scBand_int(vector<vector<int> >& query_vec,
//	 		vector<int>& query_blade_id_vec,
//	 		thrust::device_vector<int>& d_blade_data_vec,
//	 		thrust::device_vector<int>& d_blade_data_vec_endIdx,
//	 		vector<int>& d_blade_data_vec_size,
//	 		vector<int>& topk_vec, int sc_band,
//	 		vector<vector<topNode> >& _topk_results);

	 void computeTopk_dtw_scBand_float(vector<vector<float> >& query_vec,
	 		vector<int>& query_blade_id_vec,
	 		thrust::device_vector<float>& d_blade_data_vec,
	 		thrust::device_vector<int>& d_blade_data_vec_endIdx,
	 		vector<int>& d_blade_data_vec_size,
	 		vector<int>& topk_vec, int sc_band,
	 		vector<vector<topNode> >& _topk_results);

	 void computeTopk_dtwEnlb_scBand_float(vector<vector<float> >& query_vec,
	 		vector<int>& query_blade_id_vec,
	 		vector<vector<float> >& bladeData_vec,
	 		vector<int>& topk_vec,
	 		int sc_band,
	 		vector<vector<topNode> >& _topk_result);

	// template <class T>
//	 void computeTopk_dtw_scBand(vector<vector<T> >& query_vec, vector<int>& query_blade_id_vec,
//			 thrust::device_vector<T>& d_blade_data_vec,  thrust:: device_vector<int>& d_blade_data_vec_endIdx, vector<int>& d_blade_data_vec_size,
//	 		vector<int>& topk_vec, int sc_band,vector<vector<int> >& _topk_result_featureId, vector<vector<float> >& _topk_result_dist);


	 void computeTopk_dtw_scBand_ignoreStep(vector<vector<float> >& query_vec,
	 		vector<int>& query_blade_id_vec, thrust::device_vector<float>& d_blade_data_vec,
	 		thrust::device_vector<int>& d_blade_data_vec_endIdx,
	 		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
	 		vector<vector<int> >& _topk_result_featureId,
	 		vector<vector<float> >& _topk_result_dist, int ignoreStep);

//	 void computeTopk_dtw_scBand_int(vector<vector<int> >& query_vec,
//	 		vector<int>& query_blade_id_vec, thrust::device_vector<int>& d_blade_data_vec,
//	 		thrust::device_vector<int>& d_blade_data_vec_endIdx,
//	 		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
//	 		vector<vector<int> >& _topk_result_featureId,
//	 		vector<vector<float> >& _topk_result_dist);

	 void computeTopk_dtw_scBand_float(vector<vector<float> >& query_vec,
	 		vector<int>& query_blade_id_vec, thrust::device_vector<float>& d_blade_data_vec,
	 		thrust::device_vector<int>& d_blade_data_vec_endIdx,
	 		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
	 		vector<vector<int> >& _topk_result_featureId,
	 		vector<vector<float> >& _topk_result_dist);

	 void printResult(vector<vector<int> >& topk_result_idx, //output: for each queries, retun a vector of id
	 		vector<vector<float> >& topk_result_dist //output:  for each queries, retun a vector of  distance
	 		);
	 void printResult(vector<vector<topNode> >& topk_result //output: for each queries, retun a vector of id
	 		);

private:
	 //int ignore_step;//special note: igonore the first element since it is the query itself if we do not remove the query data from the dataset
};

#endif  /* GPUSCAN_H_*/

