/*
 * UtlGenie.h
 *
 *  Created on: Jul 26, 2014
 *      Author: zhoujingbo
 */

#ifndef UTLGENIE_H_
#define UTLGENIE_H_



#include <vector>
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <iostream>
#include <sstream>
using namespace std;

#include "../tools/BladeLoader.h"
#include "../tools/DataOperator/DataProcess.h"
#include "../tools/DataOperator/invListBuilder.h"
#include "../tools/DataOperator/QuerySampler.h"


class UtlGenie {
public:
	UtlGenie();
	virtual ~UtlGenie();

public:
	static string getDataFile(string fileHolder){
		stringstream ssDataFile;
		ssDataFile << fileHolder << ".csv";
		string dataFile = ssDataFile.str();
		return dataFile;
	}

	static string getQueryFile(string fileHolder, int dimensionNum, int queryNum){
		stringstream ssQueryFile;
		ssQueryFile << fileHolder << "_d" << dimensionNum << "_q" << queryNum
						<< "_dir.query";
		string queryFile = ssQueryFile.str();
		return queryFile;
	}


	static void loadDataQuery_mulBlades(string fileHolder, int fcol_start, int fcol_end, int queryNumPerBlade, int queryLen, int contStep,
			vector<vector<float> >& query_master_vec,
			vector<int>& query_blade_map,
			vector<vector<float> >& bladeData_vec
		){

		int bladeNum = fcol_end-fcol_start+1;
		string dataFile = UtlGenie::getDataFile(fileHolder);

		query_master_vec.reserve(bladeNum*queryNumPerBlade);
		query_blade_map.reserve(bladeNum*queryNumPerBlade);
		bladeData_vec.resize(bladeNum);

		cout<<"//=====loadDataQuery_mulBlades()::start loading data..."<<endl;
		DataProcess dp;
		vector<vector<float> > file_data;
		dp.ReadFileFloat_byCol(dataFile.c_str(),file_data);


		cout<<"//=====loadDataQuery_mulBlades()::finished loading data..."<<endl;

		QuerySampler qser;
		cout<<"//=====loadDataQuery_mulBlades()::start loading query..."<<endl;
		for(int i=fcol_start;i<=fcol_end;i++){
			vector<vector<float> > query;
		//	cout<<"for debu 1 bladeData_vec.size():"<<bladeData_vec.size();
			bladeData_vec[i-fcol_start] = file_data[i];
			qser.getSampleQuery_flt( bladeData_vec[i-fcol_start], queryNumPerBlade , queryLen+contStep, query);

			for(int j=0;j<query.size();j++){
				query_master_vec.push_back(query[j]);
				query_blade_map.push_back(i-fcol_start);
			}
			//cout<<"====run_CPUvsGPU_scanExp_cont()::finished loading query of column:"<<i<<endl;
		}
		cout<<"//=====loadDataQuery_mulBlades()::finished loading query..."<<endl;
	};


	//load query out from data, i.e. blade data does not contain the query information.
	static void loadDataQuery_leaveOut_mulBlades(string fileHolder, int fcol_start, int fcol_end, int queryNumPerBlade, int queryLen, //int contStep,
			vector<vector<float> >& query_master_vec,
			vector<int>& query_blade_map,
			vector<vector<float> >& bladeData_vec){

		int bladeNum = fcol_end - fcol_start + 1;
		string dataFile = UtlGenie::getDataFile(fileHolder);

		query_master_vec.reserve(bladeNum * queryNumPerBlade);
		query_blade_map.reserve(bladeNum * queryNumPerBlade);
		bladeData_vec.resize(bladeNum*queryNumPerBlade);

		cout << "//=====loadDataQuery_leaveOut_mulBlades()::start loading data..."
				<< endl;
		DataProcess dp;
		vector<vector<float> > file_data;
		dp.ReadFileFloat_byCol(dataFile.c_str(), file_data);

		cout << "//=====loadDataQuery_leaveOut_mulBlades()::finished loading data..."
				<< endl;

		QuerySampler qser;
		cout << "//=====loadDataQuery_leaveOut_mulBlades()::start loading query..."<< endl;
		for (int i = fcol_start; i <= fcol_end; i++) {

			map<uint, vector<float> > query_map;

			qser.getSampleQuery(file_data[i], queryNumPerBlade ,
					queryLen, query_map);
			int query_count=0;

			for(map<uint, vector<float> >::iterator itr=query_map.begin();itr!=query_map.end();++itr){
				query_master_vec.push_back(itr->second);
				int sensorIdx = (i - fcol_start)*queryNumPerBlade+query_count;

				query_blade_map.push_back(sensorIdx);
				bladeData_vec[sensorIdx] = file_data[i];
				int start_del = itr->first-itr->second.size();
				start_del=(start_del>0)?start_del:0;

				bladeData_vec[sensorIdx].erase(bladeData_vec[sensorIdx].begin()+start_del,
						bladeData_vec[sensorIdx].begin()+itr->first+itr->second.size());
				query_count++;

			}
			//cout<<"====run_CPUvsGPU_scanExp_cont()::finished loading query of column:"<<i<<endl;
		}

		cout << "//=====loadDataQuery_leaveOut_mulBlades()::finished loading query..."
				<< endl;

	}

	/**
		 * default with floating point data type
		 */
	static void prepareQueryFile_float(string dataFile, int fCol, string queryName,int queryNum, int dim){

		ifstream qfile(queryName.c_str());
		//if(qfile) return;

		bool rand = false;
		QuerySampler qser;
		qser.buildSampleQuery_float(dataFile, fCol,  queryName, queryNum,  dim, rand);

	}

	static void prepareGroupQueryFile_float(string dataHolder, int fcol,  int groupQueryNum, int groupQuery_item_maxlen, bool rand=false){
		QuerySampler qs;
		std::stringstream queryFileStream;
		queryFileStream <<dataHolder <<"_ql" << groupQuery_item_maxlen << "_gqn"
					<< groupQueryNum << "_group.query";
		string groupQueryFile = queryFileStream.str();
		ifstream qfile(groupQueryFile.c_str());
		if(qfile) return;
		//get query from data files
		string bladeDataFile = dataHolder+".csv";

		qs.buildSampleGroupQuery_flt(bladeDataFile, fcol, groupQueryFile,groupQueryNum, groupQuery_item_maxlen);//here!!!
	}
};

#endif /* UTLGENIE_H_ */
