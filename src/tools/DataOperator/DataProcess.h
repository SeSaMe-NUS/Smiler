/*
 * DataProcess.h
 *
 *  Created on: Dec 23, 2013
 *      Author: zhoujingbo
 */

#ifndef DATAPROCESS_H_
#define DATAPROCESS_H_

#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <cmath>

#include "UtlIndexBuilder.h"
#include "../../searcher/Scan/UtlScan.h"

using namespace std;

class DataProcess {

private:

	double min;
	double max;

public:
	DataProcess();
	virtual ~DataProcess();

private:
	vector<string> split(string& str, const char* c);
	string eraseSpace(string origin);
	int rangPartition(double v, double dw);

public:
	//vector<int>* Bucketized(vector<double>* data, double dw);
	void Bucketized(vector<double>& data, double dw, vector<int>& _dbRes);
	double getBukWidth(int bukNum);

//read from file
public:
	//used for reading data file
	template<typename F>
	void ReadFile(const char* fname, int fcol, F (*atoX)(const char *),vector<F>& _data);

	void ReadFileInt(const char* fname, int fcol,vector<int>& _data );
	void ReadFileDouble(const char* fname, int fcol, vector<double>& _data);
	void ReadFileFloat(const char* fname,int fcol, vector<float>& _data);

	//used for reading query file
	template<typename E>
	void ReadFile(const char * fname, E (*atoX)(const char *), vector<vector<E> >& _data);
	void ReadFileInt(const char * fname, vector<vector<int> >& _data);
	void ReadFileFloat(const char* fname, vector<vector<float> >& _data);
	void ReadFileDouble(const char* fname, vector<vector<double> >& _data);

	template<typename E>
	void ReadFile_byComma(const char * fname, E (*atoX)(const char *), vector<vector<E> >& _data);
	void ReadFileDouble_byComma(const char* fname, vector<vector<double> >& _data);
	void ReadFileFloat_byCol(const char* fname, vector<vector<float> >& _data);


	//vector<int>* ReadFileWidBucket(const char * fname, int fcol, double dw);
	void ReadFileWidBucket(const char * fname, int fcol, double dw, vector<int>& _dbRes);
	//vector<int>* ReadFileBucket(const char * fname, int fcol, int bukNum);
	void ReadFileBucket(const char * fname, int fcol, int bukNum, vector<int> _dbRes);

//write into binary file
public:
	void writeBinaryFile(const char * outfile, int numDim, int maxValuePerDim, map<uint, vector<int> >& im);
	//keyLow: the nuber of bits for value, the rest of the bits are left for dimensions.
	void writeInvListQueryFile(const char * outfile,
			map<uint, vector<int> > &query, int keyLow);
	void writeBinaryQueryFile(const char* outfile,map<uint, vector<int>*> &query);
	//void writeQueryFile(const char* outfile, map<uint, vector<int>*> &query);
	//template<typename H>
	//void writeQueryFile(const char* outfile,map<uint, vector<H> > &query);

	void writeQueryFile_float(const char* outfile,
			map<uint, vector<float> > &query);
	void writeQueryFile_int(const char* outfile,
			map<uint, vector<int> > &query);

	//this function is to prepare training data for for PSGP predictor
	template<typename G>
	void BuildTrainData(const char * outfile, vector<G>& data, int dim);

	void BuildTrainDataInt(const char * outfile, vector<int>& data, int dim);

	template<typename G>
	void buildTrainData(vector<G>& data,vector<vector<G> >& data_trn,int dim, int y_offset);
	void buildTrainDataFlt(vector<float>& data, vector<vector<float> >& data_trn,int dim, int y_offset);

	void run();

	void z_normalizeData(string inputFilename,  int fcol, string outputFilename, double rangeMultipler = 1);
	void z_normalizeData(string inputFilename, int fcol_start,int fcol_end, string outputFilename,double rangeMultipler=1);
	void z_normalizeData_perVector(vector<double>& data, double& mean, double& dev);
	void remove_missData(string inputFilename,  int fcol, string outputFilename);

	void printMaxMin(){
		std::cout<<" max ="<<max<<" min="<<min<<std::endl;
	}


};



#endif /* DATAPROCESS_H_ */
