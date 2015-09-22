/*
 * CPUScan.h
 *
 *  Created on: Jun 11, 2014
 *      Author: zhoujingbo
 */

#ifndef CPUSCAN_H_
#define CPUSCAN_H_
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>

#include <vector>
#include <iostream>
#include <limits.h>
#include <algorithm>
using namespace std;
#include "UtlScan.h"
#include "DistFunc.h"



class CPUScan {
public:
	CPUScan();
	virtual ~CPUScan();

public:
	void printResult(vector<vector<topNode> >& resVec) {
		cout<<"the result of CPU scan is:"<<endl;
		for (int i = 0; i < resVec.size(); i++) {
			//if(i==5){//
			cout << "print result of Query[" << i << "]" << endl;
			cout<<"start ================================"<<endl;

			for (int j = 0; j < resVec[i].size(); j++) {
				//if((j==resVec[i].size()-1)||(j==resVec[i].size()-2)){//
				resVec[i][j].print(); //<<endl;
				//}//
			}

			cout<<"end =================================="<<endl;
			//}//
		}
	}


public:
	template<class T, class DISTFUNC>
	//=============for cmputing top-k
	void CPU_computTopk(vector<vector<T> >& query, int k, vector<T> & data,
			DISTFUNC distFunc) {
		long t = 0;

		vector<vector<topNode> > resVec(query.size());
		struct timeval tim;
		gettimeofday(&tim, NULL);
		double t_start=tim.tv_sec+(tim.tv_usec/1000000.0);

		for (uint i = 0; i < query.size(); i++) {
			long start = clock();
			CPU_compTopkItem(query[i], k, data, resVec[i], distFunc); //indexVec[i], distVec[i]);

			long end = clock();
			t += (end - start);
		}
		gettimeofday(&tim, NULL);
		double t_end=tim.tv_sec+(tim.tv_usec/1000000.0);
		cout<<"running time of CPU scan is:"<<t_end-t_start<<" s"<<endl;
		cout << "the time of top-" << k << " in CPU version is:"
				<< (double) t / CLOCKS_PER_SEC << endl;
		//printResult(vector<vector<topNode> >& resVec) ;
	}

	//auxiliary function of CPU_computTopk()
	template<class T, class DISTFUNC>
	void CPU_compTopkItem(vector<T>& q, int k, vector<T> & data,
			vector<topNode>& res, DISTFUNC distFunc) {

		int dim = q.size();
		res.clear();
		res.resize(k);
		for (int r = 0; r < k; r++) {
			res[r].idx = 0;
			res[r].dis = (float) INT_MAX;
		}

		make_heap(res.begin(), res.end(), CompareTopNode());


		for (uint i = 0; i < data.size() - dim; i++) {

			float di = 0;

			di = distFunc.dist(q.data(), 0, data.data(), i, dim);

			//if smaller than maxd, replace
			if (di < res.front().dis) {
				std::pop_heap(res.begin(), res.end());
				res.pop_back();

				res.push_back(topNode(di, i));
				std::push_heap(res.begin(), res.end());
			}
		}

		std::sort_heap(res.begin(), res.end());

	}



	void computTopk_int_eu(vector<vector<int> >& query, int k,
			vector<int> & data) {


		CPU_computTopk(query, k, data, Eu_Func<int>());

	}

	void computTopk_int_dtw_scBand(vector<vector<int> >& query, int k,
			vector<int> & data, int sc_band) {

		CPU_computTopk(query, k, data, Dtw_SCBand_Func_modulus<int>(sc_band));
	}


	void computTopk_float_eu(vector<vector<float> >& query, int k,
			vector<float> & data) {


		CPU_computTopk(query, k, data, Eu_Func<float>());

	}

	void computTopk_float_dtw_scBand(vector<vector<float> >& query, int k,
			vector<float> & data, int sc_band) {

		CPU_computTopk(query, k, data, Dtw_SCBand_Func_modulus<float>(sc_band));
	}



	//=====this is special function for early termination of DTW only, it is not general function in Genei-and-Lamp-GPU

	template<class T>
			//=============for cmputing top-k
	void CPU_computTopk_Dtw_earlyStop(vector<vector<T> >& query_vec, vector<int>& query_blade_map, vector<int>& topk_vec, vector<vector<T> >& bladeData_vec,
					int sc_band) {

		vector<vector<topNode> > resVec(query_vec.size());

		for (uint i = 0; i < query_vec.size(); i++) {
			int bid = query_blade_map[i];
			CPU_compTopkItem_earlyStop(query_vec[i], topk_vec[i], bladeData_vec[bid], resVec[i],
					 Dtw_SCBand_Func_modulus<T>(sc_band), sc_band); //indexVec[i], distVec[i]);
		}

		printResult( resVec) ;//with debug purpose//
	}



		//auxiliary function of CPU_computTopk()
		template<class T, class DISTFUNC>
		void CPU_compTopkItem_earlyStop(vector<T>& q, int k, vector<T> & data,
				vector<topNode>& res, DISTFUNC distFunc, int sc_band) {

			int dim = q.size();
			res.clear();
			res.resize(k);
			for (int r = 0; r < k; r++) {
				res[r].idx = 0;
				res[r].dis = (float) INT_MAX;
			}

			make_heap(res.begin(), res.end(), CompareTopNode());
			Dtw_SCBand_LBKeogh<T> lb_keogh(sc_band);

			for (uint i = 0; i < data.size() - dim+1; i++) {//

				float lb = 0;
				lb = lb_keogh.LowerBound_keogh_byQuery(q.data(), 0, data.data(), i,
						dim);
				if (lb > res.front().dis) {

					continue;
				}
				lb = lb_keogh.LowerBound_keogh_byData(q.data(), 0, data.data(), i,
						dim);
				if (lb > res.front().dis) {

					continue;
				}

				float di = 0;

				di = distFunc.dist(q.data(), 0, data.data(), i, dim);

				//if smaller than maxd, replace
				if (di < res.front().dis) {
					std::pop_heap(res.begin(), res.end());
					res.pop_back();

					res.push_back(topNode(di, i));
					std::push_heap(res.begin(), res.end());
				}
			}

			std::sort_heap(res.begin(), res.end());

		}

	//=====

	void computTopk_int_dtw_scBand_earlyStop(vector<vector<int> >& query, int k,
			vector<int> & data, int sc_band) {


		CPU_computTopk_earlyStop(query, k, data, Dtw_SCBand_Func_modulus<int>(sc_band),sc_band);
	}

	template<class T, class DISTFUNC>
			//=============for cmputing top-k
			void CPU_computTopk_earlyStop(vector<vector<T> >& query, int k, vector<T> & data,
					DISTFUNC distFunc,int sc_band) {


				vector<vector<topNode> > resVec(query.size());

				struct timeval tim;
				gettimeofday(&tim, NULL);
				double t_start=tim.tv_sec+(tim.tv_usec/1000000.0);

				vector<vector<int> > bladeData(query.size());
				for(int i=0;i<query.size();i++){
					bladeData[i] = data;
				}

				for (uint i = 0; i < query.size(); i++) {
					CPU_compTopkItem_earlyStop(query[i], k, bladeData[i], resVec[i], distFunc, sc_band); //indexVec[i], distVec[i]);
				}

				gettimeofday(&tim, NULL);
				double t_end=tim.tv_sec+(tim.tv_usec/1000000.0);
				cout<<"running time of CPU scan is:"<<t_end-t_start<<" s"<<endl;

				//printResult(vector<vector<topNode> >& resVec) ;

			}




};

#endif /* CPUSCAN_H_ */
