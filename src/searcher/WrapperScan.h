/*
 * WrapperDTW.h
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERSCAN_H_
#define WRAPPERSCAN_H_
#include <vector>
#include <string>
#include "Scan/UtlScan.h"

using namespace std;



class WrapperScan {
public:
	WrapperScan();
	virtual ~WrapperScan();


	//====
	int runCPUEu();
	int runCpuDtw_scBand();

	int runGPUEu();
	int depressed_runCPUvsGPU_scan_singleBlade();
	int run_CPUvsGPU_scanExp();
	void run_CPUvsGPU_scanExp(vector<vector<float> >& query_vec, vector<vector<float> >& bladeData_vec,
			vector<int>& query_blade_map, int topk, int sc_band);
	void run_CPUvsGPU_scanExp(string fileHolder, int dataFileCol, int dimensionNum, int queryNum, int topk, int sc_band);

	void run_CPUvsGPU_scanExp_cont(string fileHolder,int fcol_start, int fcol_end,
			int queryNumPerBlade,int queryLen, int contStep,int top, int sc_band);
	void run_CPUvsGPU_scanExp_cont(vector<vector<float> >& query_vec, vector<vector<float> >& bladeData_vec,
				vector<int>& query_blade_map, int queryLen, int contStep, int topk, int sc_band);

	void run_CPUvsGPU_scanExp_contGroupQuery(
			string fileHolder,int fcol_start, int fcol_end,
			int queryNumPerBlade,
			int* query_item_len_vec, int query_item_num,
			int contStep,int topk, int sc_band);
	void run_CPUvsGPU_scanExp_contGroupQuery(
			vector<vector<float> >& query_master_vec,
			vector<vector<float> >& bladeData_vec,
			vector<int>& query_blade_map,
			int* query_item_len_vec, int query_item_num,
			int contStep, int topk, int sc_band);

	void run_GPU_dtwEnlb_contGroupQuery(
			vector<vector<float> >& masterQuery_vec,
			vector<vector<float> >& bladeData_vec,
			vector<int>& masterQuery_blade_map,
			int* query_item_len_vec, int query_item_num,
			int contStep, int topk, int sc_band);
};



#endif /* WRAPPERSCAN_H_ */
