/*
 * invListBuilder.cpp
 *
 *  Created on: Dec 23, 2013
 *      Author: zhoujingbo
 */

#include "invListBuilder.h"
#include <time.h>
#include <algorithm>//std::sort

using namespace std;

invListBuilder::invListBuilder() {
	// TODO Auto-generated constructor stub
	bits_for_value=24;

	assert(bits_for_value<32);

	dim = 16;

	display = false;



}

invListBuilder::invListBuilder(int k, int d){

	bits_for_value=k;
	assert(bits_for_value<32);

	dim=d;
	display = false;


}

invListBuilder::~invListBuilder() {
	// TODO Auto-generated destructor stub

}

string invListBuilder::getStringInput(string command){

	string inputFilename;
	cout << command << endl;
	cin >> inputFilename;
	return inputFilename;
}



int invListBuilder::getIntInput(string command){

	cout << command << endl;
	int columnAv;
	cin >> columnAv;
	return columnAv;
}

/**
 * get the composite key, the higher 8 bit is dimension id, and the low 24 bit is value
 * for time series CalIt2 data, the input length define the number of dimesions
 */
uint invListBuilder::getCompKey(int dim,int v){
	uint key=-1;
	int dimkey=(dim<<bits_for_value);
	key =dimkey+v;

	return key;
}

/**
 * get inverted list of sliding windows
 * _im: output result
 */
void  invListBuilder::getInvListSlidingWindow(vector<int>& data, map <uint, vector <int> >& _im){

	//map<uint,vector <int>* > *im=new map<uint,vector<int>* >;

	for(uint i=0;i<data.size()-dim+1;i++){
		for(int j=0;j<dim;j++){

			int v = data[i+j];
			uint ck = getCompKey(j,v);//form the key with dimension and value

			vector<int>& lv = _im[ck];
			lv.push_back(i);//attached this element into list
		}
	}

	//return _im;
}

/**
 * this function is for testing multiple sensors with multiple queries
 */
void invListBuilder::get_multiBlades_InvListSlidingWindow(vector<int>& data, int groupNum, int dim,  map <uint, vector <int> >& _im){

	for(uint i=0;i<data.size()-dim+1;i++){
		for(uint j=0;j<dim;j++){
			int v = data[i+j];
			for(uint g =0;g<groupNum;g++){
				int fake_dim = g*dim+j;
				uint ck = getCompKey(fake_dim,v);
				vector<int>& lv = _im[ck];
				lv.push_back(i);
			}
		}
	}
}



/**
 * get inverted list of disjoint windows
 * _im:output result
 */
void invListBuilder::getInvListDisjointWindow(vector<int>& data, map<uint, vector<int> >& _im) {

	//map<uint, vector<int>*> *im = new map<uint, vector<int>*>;

	//remove the tails, if the disjoint window is smaller than dim, we do not add them into inverted list.
	int len = (data.size()/dim)*dim;

	for (uint i = 0; i < data.size()&&i<len; i++) {

		int v = data[i];
		int dwIdx = i / dim; //id of the disjoint window
		int dimIdx = i % dim; //dimension id of this disjoint window
		uint ck = getCompKey(dimIdx, v); //form key with dimension and value

		vector<int>& lv = (_im)[ck]; //fetch the inverted list
		lv.push_back(dwIdx);
	}
	//return _im;
}


/**
 * this function is for testing multiple sensors with multiple queries
 */
void invListBuilder::get_multiBlades_InvListDisjointWindow(vector<int>& data, int groupNum, int dimIn,  map <uint, vector <int> >& _im){


	//remove the tails, if the disjoint window is smaller than dim, we do not add them into inverted list.
	int len = (data.size() / dimIn) * dimIn;

	for (uint i = 0; i < data.size() && i < len; i++) {
		int v = data[i];
		int dwIdx = i / dimIn; //id of the disjoint window
		int dimIdx = i % dimIn; //dimension id of this disjoint window
		for (uint g = 0; g < groupNum; g++) {
			int fake_dimIdx = dimIdx+g*dimIn;
			uint ck = getCompKey(fake_dimIdx, v); //form key with dimension and value
			vector<int>& lv = (_im)[ck]; //fetch the inverted list
			lv.push_back(dwIdx);
		}
	}
	//return _im;

}


/**
 * generate query without any key shifting
 */
void invListBuilder::getSampleQuery(vector<int>& data, int queryNum , map<uint, vector<int> >& _query, bool rnd){

	getSampleQuery( data,  queryNum , dim,  _query, rnd);
	//return query;
}



/**
 * generate query without any key shifting
 */
void invListBuilder::getSampleQuery(vector<int>& data, int queryNum , int queryLen, map<uint, vector<int> >& _query, bool rnd){

	//sample some query data
	int si = (data.size() - queryLen) / queryNum;
	//map<uint, vector<int>*>* query = new map<uint, vector<int>*> ();
	long time=0;
	for (int i = 0; i < queryNum; i++) {

		int st = i * si;
		if(rnd==true){
			st =st+( rand() % si);
		}

		vector<int> veci;// = new vector<int> ();
		vector<int> qi;
		qi.resize(queryLen);
		veci.resize(queryLen);
		for(int j=0;j<queryLen;j++){

			int d = data.at(st+j);
			veci[j]=d;//.push_back(d);
			qi[j]=data.at(st+j);//.push_back(data.at(st+j));
		}

		if(display){
		cout<<"query id:"<<st<<endl;
		}



		(_query)[st] = veci;
	}

}




vector<int> invListBuilder::convertQuery(vector<int> qi){
	vector<int> qr;

	for(uint i=0;i<qi.size();i++){
		int key = qi[i];
		int v = key - (i<<bits_for_value);
		qr.push_back(v);
	}

	return qr;
}

void  invListBuilder::computTopk(vector< vector <int> >& query, int k, vector<int> & data){

	long t=0;
	for(uint i=0;i<query.size();i++){
		long start = clock();
		//vector<int> qr = convertQuery(*query[i]);
		compTopkItem(query[i],  k,  data);
		long end=clock();
		t+=(end-start);
	}

	cout<<"the time of top-"<< k <<" in CPU version is:"<< (double)t / CLOCKS_PER_SEC <<endl;
}

//top3
void invListBuilder::compTopkItem(vector<int>& q, int k, vector<int> & data){
	vector<int> index;
	vector<double> dist;

	for(int r=0;r<k;r++){
		index.push_back(0);
		dist.push_back(1.0e16);
	}


	for(uint i=0;i<data.size()- dim ;i++){

		double di=0;

		for(int j=0;j<dim;j++){
			di += (q[j] - data[i+j])*(q[j] - data[i+j]);
		}

		//compute the max value in this array
		double maxd=-1; int idx=-1;
		for(int r=0;r<k;r++){
			if(dist[r] >= maxd){

				maxd=dist[r];
				idx=r;

			}
		}

		//if smaller than maxd, replace
		if(di<=maxd){
			dist[idx]=di;
			index[idx]=i;
		}
	}

}

void invListBuilder::printRes(vector<int>& q,vector<int> & data){

	cout<<"special print of query 0 from GPU"<<endl;
	int	index[]={0, 2069,1581,4413,695};// for query 3

	for(int i=0;i<5;i++){
		double di=0;
		int st=index[i];
		for(int j=0;j<dim;j++){
			di += (q[j] - data[st+j])*(q[j] - data[st+j]);
		}
		cout<<"feature id is:"<<index[i]<<" distance is:" << di<<endl;
	}

}


void invListBuilder::printCompKey(int key){

	int dim=(key>>bits_for_value);
	int v=key-(dim<<bits_for_value);
	cout<<"key is:"<<key<<endl;
	cout<<"dim is:"<<dim;
	cout<<" value is:"<<v<<endl;
}

void invListBuilder::runQuery(){

	string fName = getStringInput("The input data file name is:");
	int fCol=getIntInput("Which column is going to be used? (Start with column 0)" );
	string qName = getStringInput("The input query file name is:");
	string ft=getStringInput("input the data file type:i or d (int or double):");


	int dimIn=getIntInput("The number of dimension is:");
	int lkIn = getIntInput("The number of bits for value in key is:");
	int kInt = getIntInput("The top-k is:");

	runQuery( fName, fCol, qName, ft, dimIn, lkIn, kInt);

}


void invListBuilder::runQuery(string fName,int fCol,string qName,string ft,int dimIn,int bits_for_value,int kInt){

	dim = dimIn;
	this->bits_for_value = bits_for_value;

	DataProcess dp;
	vector<vector<int> > query;

	dp.ReadFileInt(qName.c_str(), query);

	vector<int> data;
	readDataFile(fName, fCol, ft, data);

	computTopk(query, kInt, data);


	cout << "CPU version of top-" << kInt << " is end" << endl;

}


void invListBuilder::readDataFile(string fName, int fCol,string ft, vector<int>& _data){

	//read file from raw data, and store in array data
	DataProcess dp;
	if(ft.find("i")!=string::npos) {

		dp.ReadFileInt(fName.c_str(),fCol, _data);

	} else {
		int bukNum =(int)pow(2,bits_for_value);
		dp.ReadFileBucket(fName.c_str(),fCol,bukNum, _data);
	}

//	return data;
}

void invListBuilder::runBuildIdx(){

	string fName = getStringInput("The input data file name is:");
	int fCol=getIntInput("Which column is going to be used? (Start with column 0)" );
	string ft=getStringInput("input the data file type:i or d (int or double):");
	string wt = getStringInput("input the window types to divide the time series: s(sliding windows) or d(disjoint windows");


	string outIdxName= getStringInput("The out idx file name is:");
	string queryName= getStringInput("The query file name is:");
	int queryNum=getIntInput("The number of queries is:");

	int dimIn=getIntInput("The number of dimension is:");
	int lkIn = getIntInput("The number of bits for value in key is:");

	runBuild_IdxAndQuery( fName, fCol, ft, outIdxName, queryName, queryNum,dimIn,lkIn,wt);


}

/**
 * sting ft: define the datatype, int "i" or float "f" or double "d"
 * all of them are built with buckets
 * string win: define the type of windows, "s" is for sliding windows and "d" is for disjoint window
 */
void invListBuilder::runBuild_IdxAndQuery(string fName,int fCol,string ft,string outIdxName,string queryName,int queryNum,int dimIn,int lkIn, string winType){

	dim = dimIn;
	bits_for_value=lkIn;

	vector<int> data;
	readDataFile(fName,fCol,ft, data);

	map <uint, vector <int> > im;
	if(winType.find("s")!=string::npos) {
		getInvListSlidingWindow(data, im);
		cout<<"create inverted index with sliding windows"<<endl;
	}else{
		getInvListDisjointWindow(data,im);
		cout<<"create inverted index with disjoint windows"<<endl;
	}

	int maxValuePerDim = (int)std::pow(2,bits_for_value);
	DataProcess dp;
	dp.writeBinaryFile(outIdxName.c_str(),dim,maxValuePerDim,im);

	cout<<"Inverted index has been writen to "<<outIdxName<<" !"<<endl;

	map <uint, vector <int> > query;
	getSampleQuery(data,queryNum, query);
	//dp.writeInvListQueryFile(queryName.c_str(), *query, keyLow);
	dp.writeQueryFile_int(queryName.c_str(), query);
	cout<<"queries has been writen to "<<queryName<<" !"<<endl;

}


/**
 * sting ft: define the datatype, int "i" or float "f" or double "d"
 * string win: define the type of windows, "s" is for sliding windows and "d" is for disjoint window
 * group number: the number of sensors, one sensor for one query, there are group_number duplicates, each with a window of width dimIn
 *
 * TODO:
 * In this function, we use one time series blades to simulate an index for multiple sensors and multiple group queries.
 *
 *
 */
void invListBuilder::runBuildPesudoGroupIdx(
		string fName, int fCol, string ft,
		string outIdxName, string groupQueryName,
		int bladeNum,//number of different datas, typically, one sensor generates one blade of data
		int groupQueryNum, int groupQuery_item_maxlen,
		int winDim, int bits_for_value, string winType, bool query_random) {

	dim = winDim;
	this->bits_for_value = bits_for_value;

	vector<int> data;
	readDataFile(fName, fCol, ft, data);

	map<uint, vector<int> > im;
	if (winType.find("s") != string::npos) {
		//void invListBuilder::get_group_InvListSlidingWindows(vector<int>& data, int groupNum, int dim,  map <uint, vector <int> >& _im){
		get_multiBlades_InvListSlidingWindow(data, bladeNum, winDim, im);
		cout << "create inverted index with sliding windows" << endl;
	} else {
		//void get_group_InvListDisjointWindow(vector<int>& data, int groupNum, int dim,  map <uint, vector <int> >& _im);
		get_multiBlades_InvListDisjointWindow(data, bladeNum, winDim, im);
		cout << "create inverted index with disjoint windows" << endl;
	}

	int maxValuePerDim = (int) std::pow(2, bits_for_value);
	DataProcess dp;
	int index_total_dims = bladeNum*winDim;
	dp.writeBinaryFile(outIdxName.c_str(), index_total_dims, maxValuePerDim, im);

	cout << "Inverted index has been writen to " << outIdxName << " !" << endl;

	map<uint, vector<int> > query;
	getSampleQuery(data, groupQueryNum, groupQuery_item_maxlen, query,query_random);
	//dp.writeInvListQueryFile(queryName.c_str(), *query, keyLow);
	dp.writeQueryFile_int(groupQueryName.c_str(), query);
	cout << "queries has been writen to " << groupQueryName << " !" << endl;

}



/**
 * write the query file, compose the dimensions and value into one composite key
 */
void invListBuilder::runBuildInvListQuery(string fName, int fCol, string ft, string queryName, int queryNum, int dimIn, int lkIn){

	dim = dimIn;
	bits_for_value=lkIn;

	vector<int> data;
	readDataFile(fName, fCol, ft, data);
	dim = dimIn;

	map<uint, vector<int> > query;
	getSampleQuery(data, queryNum,query);
	DataProcess dp;
	dp.writeInvListQueryFile(queryName.c_str(), query, lkIn);
	cout << "queries has been writen to " << queryName << " !" << endl;

}



