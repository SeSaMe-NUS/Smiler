/*
 * DataEngineSmiler.h
 *
 *  Created on: Sep 3, 2015
 *      Author: zhoujingbo
 */

#ifndef DATAENGINESMILER_H_
#define DATAENGINESMILER_H_

#include "../TSProcess/TSProcessor.h"
#include "../../tools/BladeLoader.h"
#include "UtlDataEngine.h"


class DataEngineSmiler  {
public:
	DataEngineSmiler();
	virtual ~DataEngineSmiler();


	void conf_bladeData(vector<vector<float> >& in_bladeData_vec);

	void setup_contGroupQuery(
			vector<vector<float> >& groupQuery_vec,
			vector<int>& groupQuery_blade_map,
			vector<int>& gq_dim_vec,
			int sc_band,int winDim);

	void retrieveTopk(
			int topk,
			int y_offset,
			vector<XYtrnPerGroup<float> >& XYtrn_vec);

	BladeLoader<float>& get_BladeLoader_reference(int bid){
			return bladeLoader_vec[bid];
	}

	void setIgnoreStep(int ignoreStep){
		//do nothing
	}

	/**
		 * 	enhancedLowerBound_sel:
	 *						0: use d2q
	 *						1: use q2d
	 *						2: use max(d2q,q2d)
		 */
	void setEnhancedLowerbound(int sel){
			tsp.setEnhancedLowerBound_sel(sel);
	}


private:
	TSProcessor<float> tsp;
	int stepCount;

	vector<int> groupQuery_blade_map;
	vector<int> gq_dim_vec;

	vector<BladeLoader<float> > bladeLoader_vec;

};

#endif /* DATAENGINESMILER_H_ */
