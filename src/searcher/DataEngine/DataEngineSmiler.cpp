/*
 * DataEngineSmiler.cpp
 *
 *  Created on: Sep 3, 2015
 *      Author: zhoujingbo
 */

#include "DataEngineSmiler.h"

DataEngineSmiler::DataEngineSmiler() :
	tsp(), stepCount(0){
	// TODO Auto-generated constructor stub

}

DataEngineSmiler::~DataEngineSmiler() {
	// TODO Auto-generated destructor stub
}


void DataEngineSmiler::conf_bladeData(vector<vector<float> >& in_bladeData_vec){
	//do nothing for sc_band, just to keep the interface the same

	tsp.conf_TSGPUManager_blade(in_bladeData_vec);

	//load into data manipulator
	this->bladeLoader_vec.clear();
	this->bladeLoader_vec.resize(in_bladeData_vec.size());
	for (int i = 0; i < in_bladeData_vec.size(); i++) {
			bladeLoader_vec[i].loadData(in_bladeData_vec[i]);
	}


}



void setup_groupQuery(vector<vector<float> > groupQuery_vec, vector<int> query_blade_map,vector<int> gq_dim_vec, int winDim,
		vector<GroupQuery_info*>& gpuQuery_info_set){

	gpuQuery_info_set.clear();
	gpuQuery_info_set.reserve(groupQuery_vec.size());

	int maxQueryLen = gq_dim_vec.back();
	int winNumPerGroup = maxQueryLen - winDim + 1;

	for (int i = 0; i < groupQuery_vec.size(); i++) {

		vector<float> gq_data(maxQueryLen);
		for (int j = 0; j < maxQueryLen; j++) {
			gq_data[j] = groupQuery_vec[i][j];
		}

		//GroupQuery_info(int groupId, int blade_id, int startQueryId, int* groupQueryDimensions_vec, int groupQuery_item_number,float* gq_data)
		GroupQuery_info* groupQuery_info = new GroupQuery_info(i, query_blade_map[i], 0,
				gq_dim_vec.data(), gq_dim_vec.size(), gq_data.data());

		gpuQuery_info_set.push_back(groupQuery_info);
	}

}


void DataEngineSmiler::setup_contGroupQuery(
			vector<vector<float> >& groupQuery_vec,
			vector<int>& groupQuery_blade_map,
			vector<int>& gq_dim_vec,
			int sc_band, int winDim){

	vector<GroupQuery_info*> groupQuery_info_Set;//this vector will be released in the TSGPUManager

	setup_groupQuery(groupQuery_vec, groupQuery_blade_map, gq_dim_vec,  winDim,
				groupQuery_info_Set);

	tsp.conf_TSGPUManager_continuousQuery(winDim,sc_band, gq_dim_vec.back(),gq_dim_vec.size(),
					groupQuery_info_Set, groupQuery_vec);

	this->groupQuery_blade_map = groupQuery_blade_map;
	this->gq_dim_vec = gq_dim_vec;

}



inline void convert_CandidateEntry_2_XYtrain(
		int groupQueryNum,int topk,
		vector<int>& queryItem_dimension_vec,
		vector<int>& groupQuery_blade_map,
		int y_offset,
		vector<BladeLoader<float> >& bloadeLoader_vec,
		host_vector<CandidateEntry>& groupQurey_topResults,
		vector<XYtrnPerGroup<float> >& XYtrn_vec){

		XYtrn_vec.clear();
		XYtrn_vec.resize(groupQueryNum);
		int queryNum_perGroup = queryItem_dimension_vec.size();

		vector<int> itemQuery_result(topk);
		for(int i=0;i<XYtrn_vec.size();i++){
			int bld_id = groupQuery_blade_map[i];
			XYtrn_vec[i].resize(queryNum_perGroup);

			for(int j=0;j<queryNum_perGroup;j++){
				int itemQuery_result_id_start = (i*queryNum_perGroup+j)*topk;
				XYtrn_vec[i].dist[j].resize(topk);

				for(int r=0;r<topk;r++){
					itemQuery_result[r]=groupQurey_topResults[itemQuery_result_id_start+r].feature_id;
					XYtrn_vec[i].dist[j][r]= groupQurey_topResults[itemQuery_result_id_start+r].dist;
				}

				bloadeLoader_vec[bld_id].retrieveXYtrn(
						itemQuery_result,queryItem_dimension_vec[j],
						y_offset,
						XYtrn_vec[i].Xtrn[j], XYtrn_vec[i].Ytrn[j]);
			}



		}


}


void DataEngineSmiler::retrieveTopk(
			int topk,
			int y_offset,
			vector<XYtrnPerGroup<float> >& XYtrn_vec){

	host_vector<CandidateEntry> groupQurey_topResults;
	host_vector<int> groupQurey_topResults_size;

	if(this->stepCount==0){
			tsp.contFirstTopk(topk,groupQurey_topResults,groupQurey_topResults_size);
	}else{

			tsp.contNextTopk(topk,groupQurey_topResults,groupQurey_topResults_size,1);
	}
	stepCount++;

	convert_CandidateEntry_2_XYtrain(
			tsp.getGroupQueryNum(), topk,
			this->gq_dim_vec,
			this->groupQuery_blade_map,
			y_offset,
			this->bladeLoader_vec,
			groupQurey_topResults,
			XYtrn_vec);

}

