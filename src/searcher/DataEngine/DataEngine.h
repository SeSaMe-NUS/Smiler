/*
 * DataEngine.h
 *
 *  Created on: Jun 24, 2014
 *      Author: zhoujingbo
 */

#ifndef DATAENGINE_H_
#define DATAENGINE_H_
#include "UtlDataEngine.h"
#include "../../tools/BladeLoader.h"


template <class T>
class DataEngine {
public:
	DataEngine(){}
	virtual ~DataEngine(){}


	virtual void conf_bladeData(vector<vector<T> >& in_bladeData_vec, int sc_band) = 0;

	virtual void retrieveTopk(vector<vector<T> >& group_query_vec,
			vector<int>& groupQuery_blade_map,
			int topk, vector<int>& groupQuery_dimension_vec,int y_offset,
			vector<XYtrnPerGroup<T> >& XYtrn_vec) {}



	virtual void retrieveTopk(vector<vector<T> >& group_query_vec,
			vector<int>& groupQuery_blade_map,
			int groupQuery_topk, vector<int>& groupQuery_dimension_vec,
			vector<vector<vector<int> > >& _groupQuery_topk_resutls_id,
			vector<vector<vector<float> > >& _groupQuery_topk_resutls_dist) {}

	virtual BladeLoader<T>& get_BladeLoader_reference(int bid) = 0;

	virtual void setIgnoreStep(int ignoreStep)=0;

};

#endif /* DATAENGINE_H_ */
