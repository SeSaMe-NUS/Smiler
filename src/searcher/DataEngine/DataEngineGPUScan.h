/*
 * DataEngineGPUScan.h
 *
 *  Created on: Jun 25, 2014
 *      Author: zhoujingbo
 */

#ifndef DATAENGINEGPUSCAN_H_
#define DATAENGINEGPUSCAN_H_

#include "DataEngine.h"

#include "../Scan/GPUScan.h"
#include "../Scan/UtlScan.h"
#include "../../tools/BladeLoader.h"
#include "UtlDataEngine.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
using namespace thrust;

#include <iostream>
using namespace std;


class DataEngine_GPUScan_helper{
public:
	void inclusiveScan(device_vector<int>& d_vec_size, device_vector<int>& d_vec_inclusive);
};


template<class T>
class DataEngine_GPUScan: public DataEngine<T> {
public:
	DataEngine_GPUScan() :
			gpuScanner() {
			sc_band = 0;
			ignoreStep=0;
	}
	virtual ~DataEngine_GPUScan() {
	}

public:

	/*
	 * conf the blade data, stored in CPU and GPU respectively
	 */
	void conf_bladeData(vector<vector<T> >& in_bladeData_vec, int sc_band) {

		//copy to host
		this->bladeData_vec.clear();
		this->bladeData_vec.resize(in_bladeData_vec.size());
		std::copy(in_bladeData_vec.begin(), in_bladeData_vec.end(),
				this->bladeData_vec.begin());

		//load into data manipulator
		this->bladeLoader_vec.clear();
		this->bladeLoader_vec.resize(in_bladeData_vec.size());
		for (int i = 0; i < in_bladeData_vec.size(); i++) {
			bladeLoader_vec[i].loadData(in_bladeData_vec[i]);
		}

		//==== copy data to device,
		//left MAX_PRE_MAX_STEP out of GPU to avoid exceeding the bounding when query the reference time series
		blade_data_vec_size.resize(bladeData_vec.size());//size of every data blade
		int totalDataSize = 0;
		for (int i = 0; i < bladeData_vec.size(); i++) {
			blade_data_vec_size[i] = (bladeData_vec[i].size());
			totalDataSize += bladeData_vec[i].size();
		}

		//caculate data blade endidx
		host_vector<int> h_blade_data_vec_size(blade_data_vec_size);

		d_blade_data_vec_endIdx = h_blade_data_vec_size;

		DataEngine_GPUScan_helper helper;
		helper.inclusiveScan(d_blade_data_vec_endIdx,d_blade_data_vec_endIdx);

		//copy to GPU global memory

		int blade_start = 0; //
		host_vector<T> h_balde_data_vec(totalDataSize);
		for (int i = 0; i < bladeData_vec.size(); i++) {
			thrust::copy(bladeData_vec[i].begin(), bladeData_vec[i].end(),
					h_balde_data_vec.begin() + blade_start);
			blade_start += bladeData_vec[i].size();
		}

		d_blade_data_vec = h_balde_data_vec;
		//==== end copy data to device

		this->sc_band = sc_band;

	}

	/**
	 * TODO:
	 * retrieve Xtrain and Ytrain accroding to to group query
	 *
	 * @param: group_query_vec:
	 * @param: groupQuery_blade_map: the relations between query and blade, i.e. which query is issued on which blades
	 * @param: groupQuery_topk: topk for all group queries
	 * @param: groupQuery_dimension_vec: different dimensions for all group queries
	 * @param: y_offset:    -- start from 0,
 	 *	 	 	 	 	    for one-step ahead prediction, y_offset = 0,
 	 *	 	 	 	 	    for multiple step ahead prediction, y_offset = mul_step-1
	 *
	 *
	 *
	 */
	void retrieveTopk(vector<vector<T> >& group_query_vec,
			vector<int>& groupQuery_blade_map,
			int topk, vector<int>& groupQuery_dimension_vec,
			int y_offset,
			vector<XYtrnPerGroup<T> >& XYtrn_vec) {

		vector<vector<vector<int> > > topk_resutls_Id;
		vector<vector<vector<float> > > topk_resutls_dist;

		retrieveTopk(group_query_vec, groupQuery_blade_map, topk,
				groupQuery_dimension_vec, topk_resutls_Id, topk_resutls_dist);

		load_groupQuery_XYtrain(topk_resutls_Id, topk_resutls_dist,
				groupQuery_dimension_vec, groupQuery_blade_map, y_offset,XYtrn_vec);

	}


	/**
	 * TODO:
	 *
	 *@param:groupQuery_topk: topk for all group queries
	 *@param:groupQuery_dimension_vec: different dimensions for all group queries
	 *@param
	 *
	 */
	void retrieveTopk(vector<vector<T> >& group_query_vec,
			vector<int>& groupQuery_blade_map,
			int groupQuery_topk, vector<int>& groupQuery_dimension_vec,
			vector<vector<vector<int> > >& _groupQuery_topk_resutls_id,
			vector<vector<vector<float> > >& _groupQuery_topk_resutls_dist) {

		vector<vector<T> > query_items_vec;
		vector<int> queryItem_blade_map;
		vector<int> queryItem_topk;
		vector<vector<int> > queryItem_topk_results_id;
		vector<vector<float> > queryItem_topk_results_dist;

		convert_groupQuery_to_queryItem(group_query_vec, groupQuery_blade_map,
				groupQuery_topk, groupQuery_dimension_vec, query_items_vec,
				queryItem_blade_map, queryItem_topk);



		gpuScanner.computeTopk_dtw_scBand_ignoreStep(query_items_vec, queryItem_blade_map,
				d_blade_data_vec, d_blade_data_vec_endIdx,
				blade_data_vec_size, queryItem_topk, sc_band,
				queryItem_topk_results_id, queryItem_topk_results_dist,ignoreStep);




		convert_queryItemResult_to_groupQueryResult(
				queryItem_topk_results_id, queryItem_topk_results_dist,
				groupQuery_topk, groupQuery_dimension_vec.size(),
				_groupQuery_topk_resutls_id,_groupQuery_topk_resutls_dist);

	}


	BladeLoader<T>& get_BladeLoader_reference(int bid){
		return bladeLoader_vec[bid];
	}
private:
	/**
	 * TODO:
	 * convert the group query information to be query items, essentially, conver 2D query meta data into 1d query meta data
	 *
	 *@param:groupQuery_topk: topk for all group queries
	 *@param:groupQuery_dimension_vec: different dimensions for all group queries
	 */
	void convert_groupQuery_to_queryItem(
			const vector<vector<T> >& group_query_vec, 	const vector<int>& groupQuery_blade_map,
			const int groupQuery_topk, const vector<int>& groupQuery_dimension_vec, //meta data for all group queries
			vector<vector<T> >& _query_items_vec,
			vector<int>& _query_items_blade_map, vector<int>& _queryItem_topk) {

		int queryItemNum_perGroup = groupQuery_dimension_vec.size();

		_query_items_vec.clear();
		_query_items_vec.resize(group_query_vec.size() * queryItemNum_perGroup);

		_query_items_blade_map.clear();
		_query_items_blade_map.resize( group_query_vec.size() * queryItemNum_perGroup);

		_queryItem_topk.clear();
		_queryItem_topk.resize(group_query_vec.size() * queryItemNum_perGroup);

		for (int i = 0; i < group_query_vec.size(); i++) {
			int groupQuery_maxDimension = groupQuery_dimension_vec[queryItemNum_perGroup - 1];
			assert(	group_query_vec[i].size() >= groupQuery_maxDimension);

			for (int j = 0; j < queryItemNum_perGroup; j++) {
				_query_items_vec[i * queryItemNum_perGroup + j].resize(groupQuery_dimension_vec[j]);

				//reserve copy, from last to front
				std::copy(group_query_vec[i].begin()+ (groupQuery_maxDimension - groupQuery_dimension_vec[j]),
						group_query_vec[i].begin()+groupQuery_maxDimension,//fixed here
						_query_items_vec[i * queryItemNum_perGroup + j].begin());

				_query_items_blade_map[i * queryItemNum_perGroup + j] =
						groupQuery_blade_map[i];
				_queryItem_topk[i * queryItemNum_perGroup + j] =
						groupQuery_topk;

			}
		}
	}

	void convert_queryItemResult_to_groupQueryResult(
			vector<vector<int> >& queryItem_topk_results_featureId,
			vector<vector<float> >& queryItem_topk_results_dist,
			int groupQuery_topk, int queryItemNum_perGroup,
			vector<vector<vector<int> > >& _groupQuery_topk_results_featureId,
			vector<vector<vector<float> > >& _groupQuery_topk_results_dist) {

		int groupQueryNum = queryItem_topk_results_featureId.size()
				/ queryItemNum_perGroup;

		_groupQuery_topk_results_featureId.resize(groupQueryNum);
		_groupQuery_topk_results_dist.resize(groupQueryNum);

		for (int i = 0; i < groupQueryNum; i++) {

			_groupQuery_topk_results_featureId[i].resize(queryItemNum_perGroup);
			_groupQuery_topk_results_dist[i].resize(queryItemNum_perGroup);

			for (int j = 0; j < queryItemNum_perGroup; j++) {

				_groupQuery_topk_results_featureId[i][j] =
						queryItem_topk_results_featureId[i * queryItemNum_perGroup + j];

				_groupQuery_topk_results_dist[i][j] =
						queryItem_topk_results_dist[i * queryItemNum_perGroup + j];
			}
		}
	}

	/**
	 * TODO:
	 * load the training data (X and Y) by the topk query result (i.e. _groupQuery_topk_resutls_id)
	 * wrap all the result into struct XYtrnPerGroup
	 *
	 * @param: y_offset:    -- start from 0,
 	 	 	 	 	 	    for one-step ahead prediction, y_offset = 0,
 	 	 	 	 	 	    for multiple step ahead prediction, y_offset = mul_step-1
	 */

	void load_groupQuery_XYtrain(const vector<vector<vector<int> > >& _groupQuery_topk_resutls_id,
			vector<vector<vector<float> > >& _groupQuery_topk_resutls_dist,
			vector<int>& queryItem_dimension_vec,
			vector<int>& groupQuery_blade_map,
			int y_offset,
			vector<XYtrnPerGroup<T> >& XYtrn_vec){

		XYtrn_vec.resize(_groupQuery_topk_resutls_id.size()); //the number of group queries
		int queryItemNum_perGroup = queryItem_dimension_vec.size();

		for (int i = 0; i < XYtrn_vec.size(); i++) {
			int bld_id = groupQuery_blade_map[i];

			XYtrn_vec[i].resize(queryItemNum_perGroup); //number of XtrainSlice and YtrainSlice for one group query

			for (int j = 0; j < queryItemNum_perGroup; j++) {

				bladeLoader_vec[bld_id].retrieveXYtrn(
						_groupQuery_topk_resutls_id[i][j],
						queryItem_dimension_vec[j],
						y_offset, XYtrn_vec[i].Xtrn[j],
						XYtrn_vec[i].Ytrn[j]);


			}
			std::copy(_groupQuery_topk_resutls_dist[i].begin(),
					_groupQuery_topk_resutls_dist[i].end(),
					XYtrn_vec[i].dist.begin());
		}

	}

	void setIgnoreStep(int ignoreStep){

		this->ignoreStep = ignoreStep;

	}
private:
	vector<vector<T> > bladeData_vec;

	vector<BladeLoader<T> > bladeLoader_vec;

	//data in GPU
	thrust::device_vector<T> d_blade_data_vec;
	thrust::device_vector<int> d_blade_data_vec_endIdx;
	vector<int> blade_data_vec_size;

	GPUScan gpuScanner;
	int ignoreStep ;//special note: igonore the first element since it is the query itself if we do not remove the query data from the dataset
	int sc_band; //for accelerate DTW
};


#endif /* DATAENGINEGPUSCAN_H_ */
