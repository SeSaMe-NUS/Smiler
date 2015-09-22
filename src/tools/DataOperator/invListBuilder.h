/*
 * invListBuilder.h
 *
 *  Created on: Dec 23, 2013
 *      Author: zhoujingbo
 */

#ifndef INVLISTBUILDER_H_
#define INVLISTBUILDER_H_



#include <assert.h>     /* assert */
#include <map>
#include <vector>
#include <list>

#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <vector>

#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;


#include "DataProcess.h"

class invListBuilder {
public:
	invListBuilder();
	invListBuilder(int k, int d);

	virtual ~invListBuilder();

private:


	//define the low bits of composited key
	//this key defines the maximum value
	uint bits_for_value;

	//the dimension of the sliding window,i.e. the length of sliding window
	int dim;

	bool display;




private:
	string getStringInput(string command);
	int getIntInput(string command);

public:

	void setKeyLow_bitsForValue(int l){ assert(l<32); bits_for_value=l; }
	void setDim(int d){dim = d;}
	void setDisplay(bool display){ 	this->display = display; }

	uint getCompKey(int dim,int v);
	void printCompKey(int key);
	void buildIndex();

	//get inverted list of sliding windows
	void getInvListSlidingWindow(vector<int>& data, map <uint, vector <int> >& _im);
	void get_multiBlades_InvListSlidingWindow(vector<int>& data, int groupNum, int dim,  map <uint, vector <int> >& _im);

	//get inverted list of disjoint windows
	void getInvListDisjointWindow(vector<int>& data, map<uint, vector<int> >& _im);
	void get_multiBlades_InvListDisjointWindow(vector<int>& data, int groupNum, int dim,  map <uint, vector <int> >& _im);


	void getSampleQuery(vector<int>& data, int queryNum, map<uint, vector<int> >& _query, bool rnd = true);
	void getSampleQuery(vector<int>& data, int queryNum , int queryLen, map<uint, vector<int> >& _query, bool rnd = false);

//	template <class T>
//	void getSeqQuery(vector<T>& data, int queryNum , vector< vector<T> >& _query);
//	template <class T>
//	void getSeqQuery(vector<T>& data, int queryNum ,int queryLen, vector< vector<T> >& _query);


	void computTopk(vector< vector <int> >& query, int k, vector<int> & data);
	void compTopkItem(vector<int>& q, int k, vector<int> & data);
	void printRes(vector<int>& q,vector<int> & data);

	void readDataFile(string fName, int fCol,string ft, vector<int>& _data);
	vector<int> convertQuery(vector<int> qi);

	void runBuildIdx();
	void runQuery();
	void runQuery(string fName,int fCol,string qName,string ft,int dimIn,int bits_for_value,int kInt);

	void runBuild_IdxAndQuery(string fName,int fCol,string ft, string outIdxName,string queryName,int queryNum,int dimIn,int lkIn,string winType="d");

	void runBuildPesudoGroupIdx(
			string fName, int fCol, string ft,
			string outIdxName, string groupQueryName,
			int bladeNum,//number of different datas, typically, one sensor generates one blade of data
			int groupQueryNum, int max_groupQueryLen,
			int winDim, int bits_for_value, string winType = "d",  bool query_random = true);

	//void buildSampleQuery_float(string fName, int fCol, string queryName,int queryNum, int dimIn, bool rand = true);
	void runBuildInvListQuery(string fName, int fCol, string ft, string queryName, int queryNum, int dimIn, int lkIn);


};

#endif /* INVLISTBUILDER_H_ */
