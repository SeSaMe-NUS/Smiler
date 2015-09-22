/*
 * BladeLoader.h
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */
#ifndef BLADELOADER_H_
#define BLADELOADER_H_

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
using namespace std;

#include "DataOperator/DataProcess.h"

template<class T>
class BladeLoader {
public:
	BladeLoader() {

	}

	virtual ~BladeLoader() {

	}

	std::vector<T> data;

public:
	int size(){
		return data.size();
	}
	//load data from files
	void loadData(const char* dataName, const int col,
			T (*atoX)(const char *)) {
		DataProcess dp;
		dp.ReadFile(dataName, col, (*atoX), data);
	}

	void loadDataInt(const char * dataName, const int col) {
		DataProcess dp;
		dp.ReadFileInt(dataName, col, data);
	}
	void loadDataDouble(const char* dataName, const int col) {
		DataProcess dp;
		dp.ReadFileDouble(dataName, col, data);
	}

	void loadDataFloat(const char* dataName, const int col){
		DataProcess dp;
		dp.ReadFileFloat(dataName,col,data);
	}

	void loadData(vector<T>& data){
		this->data.clear();
		this->data.resize(data.size());
		std::copy(data.begin(),data.end(),this->data.begin());
	}

	/**
	 *@paramm: y_offset-- start from 0,
 	 *	 	 	 	 	    for one-step ahead prediction, y_offset = 0,
 	 *	 	 	 	 	    for multiple step ahead prediction, y_offset = mul_step-1
	 */
	void retrieveXYtrn(const vector<int>& st, const int Xdim, const int y_offset,
			vector<vector<T> >& _Xtrn, vector<T>& _Ytrn) {

		retrieveTS(st, Xdim, _Xtrn);
		retrieveVertical(st, Xdim+y_offset, _Ytrn);

	}

	/**
	 * retrieve X and Y for training, note that tsIdx means the the same with different dimensions.
	 */
	void retrieveXYtrn(const vector<vector<int> >& tsIdx,
			const vector<int>& Lvec, vector<vector<vector<T> > >& _XtrnRtr,
			vector<vector<T> >& _YTrnRtr) {


		retrieveXYtrn( tsIdx,
					 Lvec, 0,  _XtrnRtr,
					 _YTrnRtr);
	}

	/**
	 * TODO:
	 * retrieve Xtrn and Ytrn from historical data:
	 *
	 * @paramm: y_offset-- start from 0,
 	 	 	 	 	 	    for one-step ahead prediction, y_offset = 0,
 	 	 	 	 	 	    for multiple step ahead prediction, y_offset = mul_step-1
	 */
	void retrieveXYtrn(const vector<vector<int> >& tsIdx,
			const vector<int>& Lvec, int y_offset, vector<vector<vector<T> > >& _XtrnRtr,
			vector<vector<T> >& _YTrnRtr){

		retrieveTS(tsIdx, Lvec, _XtrnRtr);
		retrieveYNextTrn(tsIdx, Lvec, y_offset, _YTrnRtr);

	}

	void retrieveVertical(const vector<int>& st, const int offset,
			vector<T>& _res) {
		retrieveVertical(st.data(), st.size(), offset, _res);

	}

	/**Lvec:records the different dimensions for the same query
	 *
	 * _YNextTrn (return result): the next Y train data from bldLoader
	 */
	void retrieveYNextTrn(const vector<vector<int> >& resIdx,
			const vector<int> & Lvec, const int s,
			vector<vector<T> >& _YNextTrn) {

		for (int i = 0; i < resIdx.size(); i++) {

			vector<T> YNextTrnSlice;
			retrieveVertical(resIdx[i], Lvec[i] + s, YNextTrnSlice);
			_YNextTrn.push_back(YNextTrnSlice);

		}

	}

	/**
	 * retrieve a vertical slice from data
	 * st: indicate the starting position of the time series
	 * st_num: size of st
	 * offset: offset from starting position in st
	 *
	 * _res: return result
	 */
	void retrieveVertical(const int* st, const int st_num, const int offset,
			vector<T>& _res) {

		_res.resize(st_num);

		for (int i = 0; i < st_num; i++) {
			int startIdx = st[i];

			assert((startIdx + offset)<data.size());
			_res[i] = (data)[startIdx + offset];



		}

	}

	/**
	 *
	 */
	void retrieveTS(const vector<vector<int> >& resIdx, const vector<int>& Lvec,
			vector<vector<vector<T> > >& _Xtrn) {

		_Xtrn.clear();
		_Xtrn.resize(Lvec.size());
		for (int i = 0; i < resIdx.size(); i++) {
			retrieveTS(resIdx[i], Lvec[i], _Xtrn[i]);
		}

	}

	/**
	 *
	 * query a set of time series from blade,
	 * st: the starting point in the time series
	 * len: the length of each query result
	 *
	 * res: the a array record the corresponding time series
	 *
	 */
	void retrieveTS(const vector<int>& st, const int len,
			vector<vector<T> >& _res) {
		retrieveTS(st.data(), st.size(), len, _res);
	}

	/**
	 * query a set of time series from blade,
	 * st: the starting point in the time series
	 * num: the number of results, i.e. the length of st
	 * len: the length of each query result, i.e. the length of each time series
	 *
	 *
	 * return: a array record the corresponding time series
	 */
	void retrieveTS(const int* st, const int num, const int len,
			vector<vector<T> >& _res) {
		_res.clear();
		_res.resize(num);


		for (int i = 0; i < num; i++) {
			int strIdx = st[i];
			_res[i].resize(len);

			for (int j = 0; j < len; j++) {
				assert((strIdx+j)<data.size());
				T item = data[strIdx + j];
				_res[i][j] = item;


			}
		}

	}

	std::vector<T> getData() {
		return data;
	}


};

#endif /* BLADELOADER_H_ */
