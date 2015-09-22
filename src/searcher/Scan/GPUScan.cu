/*
 * GPUScan.cpp
 *
 *  Created on: Jun 16, 2014
 *      Author: zhoujingbo
 */
#include <sys/time.h>
#include "GPUScan.h"

#include <iostream>
using namespace std;

#include "TemplateFunctions/GPUScanFunctions.h"
#include "DistFunc.h"
#include "UtlScan.h"
#include "../DataEngine/UtlDataEngine.h"

#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
using namespace thrust;

GPUScan::GPUScan() {
	// TODO Auto-generated constructor stub
}

GPUScan::~GPUScan() {
	// TODO Auto-generated destructor stub
}

void convterIntToFloat(vector<vector<int> >& query,
		vector<vector<int> >& bladeData, vector<vector<float> >& query_flt,
		vector<vector<float> >& bladeData_flt) {
	//convert to float
	bladeData_flt.resize(bladeData.size());
	for (int i = 0; i < bladeData.size(); i++) {
		bladeData_flt[i].resize(bladeData[i].size());
		for (int j = 0; j < bladeData[i].size(); j++) {
			bladeData_flt[i][j] = (float) bladeData[i][j];
		}
	}

	query_flt.resize(query.size());
	for (int i = 0; i < query.size(); i++) {
		query_flt[i].resize(query[i].size());
		for (int j = 0; j < query[i].size(); j++) {
			query_flt[i][j] = (float) query[i][j];
		}
	}

}

void printResult(vector<vector<topNode> >& topk_result_idx) {

	cout << "the result of GPU scan is:" << endl;
	for (int i = 0; i < topk_result_idx.size(); i++) {
		cout << "query item [" << i << "]" << endl;
		for (int j = 0; j < topk_result_idx[i].size(); j++) {
			cout << "query item [" << i << "] result " << j << ":"
					<< topk_result_idx[i][j].idx << " dist:"
					<< topk_result_idx[i][j].dis << endl;
		}
		cout << endl;
	}
}

template<class E>
void computTopk_eu(vector<vector<E> >& query, int k,
		vector<E> & data) {

	vector<int> query_blade_id(query.size(), 0);

	vector<vector<E> > bladeData;
	bladeData.push_back(data);
	vector<vector<topNode> > topk_result_idx;

	vector<int> topk_vec(query.size(), k);

	GPU_computeTopk(query, query_blade_id, bladeData, topk_vec, Eu_Func<E>(),
			topk_result_idx);

	//printResult( topk_result_idx);

}



//void GPUScan::computTopk_eu_int(vector<vector<int> >& query, int k,
//		vector<int> & data) {
//
//	computTopk_eu( query,  k, data);
//	//printResult( topk_result_idx);
//}

void GPUScan::computTopk_eu_float(vector<vector<float> >& query, int k,
		vector<float> & data) {

	computTopk_eu( query,  k, data);
	//printResult( topk_result_idx);
}



template<class T>
void computTopk_dtw_scBand(vector<vector<T> >& query, int k,
		vector<T> & data, int sc_band) {

	vector<int> query_blade_id(query.size(), 0);
	vector<vector<T> > bladeData(query.size());

	for (int i = 0; i < query.size(); i++) {
		bladeData[i] = data;
		query_blade_id[i] = i;
	}

	vector<vector<topNode> > topk_result;
	vector<int> topk_vec(query.size(), k);

	struct timeval tim;
	gettimeofday(&tim, NULL);
	double t_start = tim.tv_sec + (tim.tv_usec / 1000000.0);

	GPU_computeTopk(query, query_blade_id, bladeData, topk_vec,
			Dtw_SCBand_Func_modulus<T>(sc_band), topk_result);


	gettimeofday(&tim, NULL);
	double t_end = tim.tv_sec + (tim.tv_usec / 1000000.0);
	cout << "running time of GPU scan is:" << t_end - t_start << " s" << endl;

	//printResult( topk_result);
}
//
//void GPUScan::computTopk_dtw_scBand_int(vector<vector<int> >& query, int k,
//		vector<int> & data, int sc_band){
//
//	computTopk_dtw_scBand( query,  k,
//			 data,  sc_band);
//}

void GPUScan::computTopk_dtw_scBand_float(vector<vector<float> >& query, int k,
		vector<float> & data, int sc_band){

	computTopk_dtw_scBand( query,  k,
			 data,  sc_band);
}





//template<class T>
////dtw computation without scband constraints
//void computTopk_dtw_modulus(vector<vector<T> >& query_vec,
//				vector<int>& query_blade_id_vec, vector<vector<T> >& bladeData_vec,
//				vector<int>& topk_vec,
//				vector<vector<topNode> >& _topk_resutls){
//
//	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
//			Dtw_Func_modulus<T>(), _topk_resutls //output
//				);
//}


void GPUScan::computTopk_dtw_modulus_float(vector<vector<float> >& query_vec,
				vector<int>& query_blade_id_vec, vector<vector<float> >& bladeData_vec,
				vector<int>& topk_vec,
				vector<vector<topNode> >& _topk_resutls){

	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
				Dtw_Func_modulus<float>(), _topk_resutls //output
					);

}

//
//void GPUScan::computTopk_dtw_modulus_int(vector<vector<int> >& query_vec,
//				vector<int>& query_blade_id_vec, vector<vector<int> >& bladeData_vec,
//				vector<int>& topk_vec,
//				vector<vector<topNode> >& _topk_resutls){
//
//	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
//				Dtw_Func_modulus<int>(), _topk_resutls //output
//					);
//
//}


//template<class T>
//void computTopk_dtw_scBand_fullWM(vector<vector<T> >& query_vec,
//		vector<int>& query_blade_id_vec, vector<vector<T> >& bladeData_vec,
//		vector<int>& topk_vec, int sc_band,
//		vector<vector<topNode> >& _topk_resutls) {
//
//	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
//			Dtw_SCBand_Func_full<T>(sc_band), _topk_resutls //output
//			);
//
//}

//
//void GPUScan::computTopk_dtw_scBand_fullWM_int(vector<vector<int> >& query_vec,
//		vector<int>& query_blade_id_vec, vector<vector<int> >& bladeData_vec,
//		vector<int>& topk_vec, int sc_band,
//		vector<vector<topNode> >& _topk_resutls) {
//
//	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
//			Dtw_SCBand_Func_full<int>(sc_band), _topk_resutls //output
//			);
//
//}

void GPUScan::computTopk_dtw_scBand_fullWM_float(vector<vector<float> >& query_vec,
		vector<int>& query_blade_id_vec, vector<vector<float> >& bladeData_vec,
		vector<int>& topk_vec, int sc_band,
		vector<vector<topNode> >& _topk_resutls) {

	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
			Dtw_SCBand_Func_full<float>(sc_band), _topk_resutls //output
			);

}



template<class T>
void computTopk_dtw_scBand(vector<vector<T> >& query_vec,
		vector<int>& query_blade_id_vec,
		vector<vector<T> >& bladeData_vec,
		vector<int>& topk_vec,
		int sc_band,
		vector<vector<topNode> >& _topk_result) {

	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
			Dtw_SCBand_Func_modulus<T>(sc_band), _topk_result //output
			);
	//printResult(_topk_result);
}


//
//void GPUScan::computTopk_dtw_scBand_int(vector<vector<int> >& query_vec,
//		vector<int>& query_blade_id_vec,
//		vector<vector<int> >& bladeData_vec,
//		vector<int>& topk_vec,
//		int sc_band,
//		vector<vector<topNode> >& _topk_result) {
//
//	computTopk_dtw_scBand(
//			 query_vec,
//			 query_blade_id_vec,
//			 bladeData_vec,
//			 topk_vec,
//			 sc_band,
//			 _topk_result);
//	//printResult(_topk_result);
//}


void GPUScan::computTopk_dtw_scBand_float(vector<vector<float> >& query_vec,
		vector<int>& query_blade_id_vec,
		vector<vector<float> >& bladeData_vec,
		vector<int>& topk_vec,
		int sc_band,
		vector<vector<topNode> >& _topk_result) {

	computTopk_dtw_scBand( query_vec,
			 query_blade_id_vec,
			 bladeData_vec,
			 topk_vec,
			 sc_band,
			_topk_result);
	//printResult(_topk_result);
}


template <class T>
void computeTopk_dtwEnlb_scBand(vector<vector<T> >& query_vec,
		vector<int>& query_blade_id_vec,
		vector<vector<T> >& bladeData_vec,
		vector<int>& topk_vec,
		int sc_band,
		vector<vector<topNode> >& _topk_result) {

	GPU_computeTopk(query_vec, query_blade_id_vec, bladeData_vec, topk_vec,
			DtwEnlb_SCBand_Func<T>(sc_band), _topk_result //output
			);
	//printResult(_topk_result);
}


void GPUScan::computeTopk_dtwEnlb_scBand_float(vector<vector<float> >& query_vec,
		vector<int>& query_blade_id_vec,
		vector<vector<float> >& bladeData_vec,
		vector<int>& topk_vec,
		int sc_band,
		vector<vector<topNode> >& _topk_result) {

	computeTopk_dtwEnlb_scBand( query_vec,
			 query_blade_id_vec,
			 bladeData_vec,
			 topk_vec,
			 sc_band,
			_topk_result);
	//printResult(_topk_result);
}


template<class T>
void computeTopk_dtw_scBand(
		vector<vector<T> >& query_vec,
		vector<int>& query_blade_id_vec,
		device_vector<T>& d_blade_data_vec,
		device_vector<int>& d_blade_data_vec_endIdx,
		vector<int>& d_blade_data_vec_size,
		vector<int>& topk_vec, int sc_band,
		vector<vector<topNode> >& _topk_results) {

	GPU_computeTopk(query_vec,
			query_blade_id_vec,
			d_blade_data_vec,
			d_blade_data_vec_endIdx,
			d_blade_data_vec_size,
			topk_vec,
			Dtw_SCBand_Func_modulus<T>(sc_band),
			_topk_results //output
			);
}

//void GPUScan::computeTopk_dtw_scBand_int(
//		vector<vector<int> >& query_vec,
//		vector<int>& query_blade_id_vec,
//		device_vector<int>& d_blade_data_vec,
//		device_vector<int>& d_blade_data_vec_endIdx,
//		vector<int>& d_blade_data_vec_size,
//		vector<int>& topk_vec, int sc_band,
//		vector<vector<topNode> >& _topk_results) {
//
//	GPU_computeTopk(query_vec,
//			query_blade_id_vec,
//			d_blade_data_vec,
//			d_blade_data_vec_endIdx,
//			d_blade_data_vec_size,
//			topk_vec,
//			Dtw_SCBand_Func_modulus<int>(sc_band),
//			_topk_results //output
//			);
//
//}

void GPUScan::computeTopk_dtw_scBand_float(
		vector<vector<float> >& query_vec,
		vector<int>& query_blade_id_vec,
		device_vector<float>& d_blade_data_vec,
		device_vector<int>& d_blade_data_vec_endIdx,
		vector<int>& d_blade_data_vec_size,
		vector<int>& topk_vec, int sc_band,
		vector<vector<topNode> >& _topk_results) {


	GPU_computeTopk(query_vec,
			query_blade_id_vec,
			d_blade_data_vec,
			d_blade_data_vec_endIdx,
			d_blade_data_vec_size,
			topk_vec,
			Dtw_SCBand_Func_modulus<float>(sc_band),
			_topk_results //output
			);

}


/**
 * TODO:
 * retreive topk from data set
 *
 * special note: igonore the first element since it is the query itself if we do not remove the query data from the dataset
 */

template<class T>
void computeTopk_dtw_scBand(vector<vector<T> >& query_vec,
		vector<int>& query_blade_id_vec, device_vector<T>& d_blade_data_vec,
		device_vector<int>& d_blade_data_vec_endIdx,
		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
		vector<vector<int> >& _topk_result_featureId,
		vector<vector<float> >& _topk_result_dist, int ignore_step) {

	vector<vector<topNode> > topk_results;

	if(ignore_step>=1){
	vector<int> topk_vec_ignoreStep (topk_vec.size());
	for(int i=0;i<topk_vec_ignoreStep.size();i++){
		topk_vec_ignoreStep[i] = topk_vec[i] + ignore_step;
	}

	GPU_computeTopk(query_vec, query_blade_id_vec, d_blade_data_vec,
			d_blade_data_vec_endIdx, d_blade_data_vec_size,
			topk_vec_ignoreStep,
			Dtw_SCBand_Func_modulus<T>(sc_band),
			topk_results //output
			);

	} else{

	GPU_computeTopk(query_vec,
			query_blade_id_vec,
			d_blade_data_vec,
			d_blade_data_vec_endIdx,
			d_blade_data_vec_size,
			topk_vec,//topk_vec_ignoreStep,
			Dtw_SCBand_Func_modulus<T>(sc_band),
			topk_results //output
			);



	}

	_topk_result_featureId.clear();
	_topk_result_featureId.resize(topk_results.size());

	_topk_result_dist.clear();
	_topk_result_dist.resize(topk_results.size());


	for (int i = 0; i < topk_results.size(); i++) {
		_topk_result_featureId[i].resize(topk_results[i].size()-ignore_step);
		_topk_result_dist[i].resize(topk_results[i].size()-ignore_step);
		for (int j = ignore_step; j < topk_results[i].size(); j++) {
			_topk_result_featureId[i][j-ignore_step] = topk_results[i][j].idx;
			_topk_result_dist[i][j-ignore_step] = topk_results[i][j].dis;
		}
	}


}


//special note: igonore the first element since it is the query itself if we do not remove the query data from the dataset
//template<class T>
void GPUScan::computeTopk_dtw_scBand_ignoreStep(vector<vector<float> >& query_vec,
		vector<int>& query_blade_id_vec, device_vector<float>& d_blade_data_vec,
		device_vector<int>& d_blade_data_vec_endIdx,
		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
		vector<vector<int> >& _topk_result_featureId,
		vector<vector<float> >& _topk_result_dist, int ignoreStep) {

	 computeTopk_dtw_scBand( query_vec,
			 query_blade_id_vec,  d_blade_data_vec,
			 d_blade_data_vec_endIdx,
			 d_blade_data_vec_size,  topk_vec,  sc_band,
			 _topk_result_featureId,
			 _topk_result_dist, ignoreStep);


	 //this->printResult(_topk_result_featureId, //output: for each queries, retun a vector of id
	//		 _topk_result_dist //output:  for each queries, retun a vector of  distance
	 //		);//for with debug purpose


}


//void GPUScan::computeTopk_dtw_scBand_int(vector<vector<int> >& query_vec,
//		vector<int>& query_blade_id_vec, device_vector<int>& d_blade_data_vec,
//		device_vector<int>& d_blade_data_vec_endIdx,
//		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
//		vector<vector<int> >& _topk_result_featureId,
//		vector<vector<float> >& _topk_result_dist){
//
//	 computeTopk_dtw_scBand( query_vec,
//			 query_blade_id_vec, d_blade_data_vec,
//			 d_blade_data_vec_endIdx,
//			 d_blade_data_vec_size,  topk_vec,  sc_band,
//			 _topk_result_featureId,
//			 _topk_result_dist,  0);
//}

void GPUScan::computeTopk_dtw_scBand_float(vector<vector<float> >& query_vec,
		vector<int>& query_blade_id_vec, device_vector<float>& d_blade_data_vec,
		device_vector<int>& d_blade_data_vec_endIdx,
		vector<int>& d_blade_data_vec_size, vector<int>& topk_vec, int sc_band,
		vector<vector<int> >& _topk_result_featureId,
		vector<vector<float> >& _topk_result_dist){

	 computeTopk_dtw_scBand( query_vec,
				 query_blade_id_vec, d_blade_data_vec,
				 d_blade_data_vec_endIdx,
				 d_blade_data_vec_size,  topk_vec,  sc_band,
				 _topk_result_featureId,
				 _topk_result_dist,  0);

}



void GPUScan::printResult(vector<vector<int> >& topk_result_idx, //output: for each queries, retun a vector of id
		vector<vector<float> >& topk_result_dist //output:  for each queries, retun a vector of  distance
		) {

	cout << "output of the the k nearest neighbor search for each query ...... "
			<< endl;
	for (int i = 0; i < topk_result_idx.size(); i++) {
		cout << "+++++++++++++++++++++++++++++++++++++++++++" << endl;
		cout << "query " << i << " is:" << endl;
		printf("order	id	distance\n");
		for (int j = 0; j < topk_result_idx[i].size(); j++) {
			printf("[%d]	%d	%f\n", j, topk_result_idx[i][j],
					topk_result_dist[i][j]);

		}
		cout << endl << endl;
	}
}

void GPUScan::printResult(vector<vector<topNode> >& topk_result //output: for each queries, retun a vector of id
		) {

	cout << "output of the the k nearest neighbor search for each query ...... "
			<< endl;
	for (int i = 0; i < topk_result.size(); i++) {

		//if(i==0||i==1){//
		cout << "+++++++++++++++++++++++++++++++++++++++++++" << endl;
		cout << "query " << i << " is:" << endl;
		printf("order	id	distance\n");
		for (int j = 0; j < topk_result[i].size(); j++) {
			printf("[%d]	%d	%f\n", j, topk_result[i][j].idx,
					topk_result[i][j].dis);

		}
		cout << endl << endl;
		//}//
	}
}


//template class GPUScan<int> ;
//template class GPUScan<float>;

