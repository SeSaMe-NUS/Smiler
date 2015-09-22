/*
 * QuerySampler.h
 *
 *  Created on: Aug 6, 2014
 *      Author: zhoujingbo
 */

#ifndef QUERYSAMPLER_H_
#define QUERYSAMPLER_H_






#include <string>
#include <iostream>
#include <sstream>
using namespace std;


#include "DataProcess.h"

/**
 * NOTE: 1. max step ahead should be smaller than MAX_MUL_STEP
 */
class QuerySampler {
public:
	QuerySampler(){
		display = false;
	}
	virtual ~QuerySampler(){

	}



public:


	/**
	 * generate query without any key shifting
	 */
	template <class T>
	void getSampleQuery(vector<T>& data, int queryNum , int queryLen, map<uint, vector<T> >& _query, bool rnd=false){

		//sample query data
		//int si = (data.size() - 2*queryLen-1) / queryNum;//for sigmod1stSubmit branch
		int si = (data.size() - queryLen-1) / queryNum;
		//map<uint, vector<int>*>* query = new map<uint, vector<int>*> ();

		for (int i = 0; i < queryNum; i++) {

			int st = (i+1) * si;
			if(rnd==true){
				st =st+( rand() % si);
			}

			vector<T> veci;// = new vector<int> ();
			vector<T> qi;
			qi.resize(queryLen);
			veci.resize(queryLen);
			for(int j=0;j<queryLen;j++){

				T d = data.at(st+j);
				veci[j]=d;//.push_back(d);
				qi[j]=data.at(st+j);//.push_back(data.at(st+j));
			}

			if(display){
			cout<<"query id:"<<st<<endl;
			}

			(_query)[st] = veci;
		}

	}


	void getSampleQuery_flt(vector<float>& data, int queryNum , int queryLen, vector<vector<float> >& _query, bool rnd=false){

		map<uint, vector<float> > query_map;
		getSampleQuery( data,  queryNum ,  queryLen,  query_map,  rnd);

		_query.clear();
		_query.reserve(query_map.size());

		for(std::map<uint, vector<float> >::iterator itr=query_map.begin();itr!=query_map.end();++itr){
			_query.push_back(itr->second);
		}

	}


	int buildSampleGroupQuery_flt(string bladeDataFile, int fcol, string groupQueryFile,  int groupQueryNum, int groupQuery_item_maxlen, bool rand=false){

		vector<float> data;
		DataProcess dp;

		dp.ReadFileFloat(bladeDataFile.c_str(),fcol,data);

		map<uint, vector<float> > query;
		getSampleQuery(data, groupQueryNum, groupQuery_item_maxlen, query,rand);
		dp.writeQueryFile_float(groupQueryFile.c_str(), query);
		cout << "queries has been writen to " << groupQueryFile << " !" << endl;

		return 0;
	}



	/**
	 * 	ilB.runBuildQuery("data/calit2/CalIt2_7.csv",3,"i","data/calit2/CalIt2_7_d64_q16_dir.query",16,64,true);
	 * write the query file, without composing the dimensions and value
	 *
	 * random select queryNum points as query data
	 *
	 * inputparameter:
	 * fname: input data file
	 * fcol: column of input data file
	 * ft: data type (i, integer, or f, float)
	 * queryName:out put file for query
	 * queryNum: number of queries
	 * dimIn: number of dimension for time series(length of backward window)
	 * rand:get sample query by randam selection. if false, we always choose the first queryNum windows as query
	 */
	void buildSampleQuery_float(string fName, int fCol, string queryName,int queryNum, int dimIn,bool rand){

		if(display){
		cout<<"fname:"<<fName<<endl;
		}
		DataProcess dp;

		vector<float> data;
		dp.ReadFileFloat(fName.c_str(),fCol,data);

		map <uint, vector <float> > query;

		getSampleQuery(data,queryNum,dimIn, query, rand);

		dp.writeQueryFile_float(queryName.c_str(), query);
		if(display){
		cout<<"queries with float point data type has been written to "<<queryName<<" !"<<endl;
		}
	}



	template<class T>
	void getSeqQuery(vector<T>& data, int queryNum, int queryLen,
			vector<vector<T> >& _query, int start) {
		_query.resize(queryNum);
		for (int i = 0; i < queryNum; i++) {
			//int st = i * si;
			vector<T> veci; // = new vector<int> ();
			vector<T> qi;
			qi.resize(queryLen);
			veci.resize(queryLen);
			for (int j = 0; j < queryLen; j++) {
				T d = data.at(i + j + start);
				veci[j] = d; //.push_back(d);
				qi[j] = data.at(i + j + start); //.push_back(data.at(st+j));
			}
			if (display) {
				cout << "query start pos:" << i + start << endl;
				for (int k = 0; k < queryLen; k++) {
					cout << " " << veci[k];
				}
				cout << endl;
			}
			(_query)[i] = veci;
		}
	}


public:
	bool display;



};

#endif /* QUERYSAMPLER_H_ */
