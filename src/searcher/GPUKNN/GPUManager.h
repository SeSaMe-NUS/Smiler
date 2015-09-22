/*
 * GPUManager.h
 *
 *  Created on: Apr 7, 2014
 *      Author: yuxin
 */

#ifndef GPUMANAGER_H_
#define GPUMANAGER_H_

#include "generalization.h"


using namespace std;
using namespace thrust;


class GPUManager {


public:
	GPUManager( );
	GPUManager( GPUSpecification& );
	GPUManager( GPUSpecification& query_spec, int rand_invert_list_size );

	/* clear all gpu memory */
	~GPUManager();

	//to record the time spent on those major steps in processing the bi-directional query
	vector<double> exec_time;
	/*read query from file, the query is a list of vectors, i.e. no additional informatino is provided such as weight and bound*/
	void readQueryFromFile(string queryFileName, vector<GpuWindowQuery>& query_set);
	void conf_GPUManager_GPUSpecification(GPUSpecification& gpu_spec);
	/* setup a new batch of query */
	void init_GPU_query(vector<GpuWindowQuery>& query_set);
	void init_dimensionNum_perQuery( int dimensionNumPerQuery ){init_dev_dimNumPerQuery_constMem(dimensionNumPerQuery);}

	void init_parameters_default();
	/*
	 run this batch of query which just initialized, should not be mixed with point_query
	 threshold is for count, if count > threshold, then result will be considered

	  each query take one block (kernel), therefore this function is query oriented parallel computing, this is an old version
	 */

	bool bi_direction_query_KernelPerDim (int threshold, int topKValue, vector<Result>& _result );

	/*
	 this method should be only call once, then no more calls including bi_direciton query should be made
	 threshold is for count, if count > threshold, then result will be considered
	 */
	void point_query(vector<Result>& result, int threshold);

	/*-------------------------------------------------------------------------------- */
	//Open API
	template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
	void dev_BidirectionExpansion(
			KEYWORDMAP keywordMap,
			LASTPOSMAP lastPosMap,
			DISTFUNC distFunc);


	/* ---------------------------------- debug purpose--------------------------------------- */
	void print_query();
	void print_query_doucment_mapping();
	void print_invert_list();
	// void print_seperate_line();
	void print_result_vec(vector<Result>& result_vec);
	void printPrunStatistics();


	// random generate inverted list
	void rand_inv_list();
	// random generate query
	void rand_query(int query_num);

	// retrieve invert list given the directory
	// void get_invert_list_from_file(string invert_list_dir,string invert_list_idx_file);

	// retrieve invert list from binary given the filename, and read the number of dimension and max value per
	//dimension from the binary file
	void get_invert_list_from_binary_file(string filenames, int& _numberOfDim,  int& _maxValuePerDim, int& _maxFeatureID);


	// clear memory to setup new round of query search
	void clear_query_memory();
	void set_query_DefaultDisFuncType(float p) {default_disfuncType=p; };
	void set_query_DefaultDistBound(float up, float down){default_upwardDistBound = up; default_downwardDistBound = down;}
	void update_windowQuery_entryAndTable(WindowQueryInfo* host_queryInfo, int d_query_info_id);
	void update_windowQuery_entryAndTable(GpuWindowQuery& windowQuery);
	void update_QueryInfo_entry_upperAndLowerBound(int queryId, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist);

	void runTest();
	int getMaxFeatureNumber() const { return invert_list_spec_host.maxFeatureNumber;}
	int getSumQueryDims() const {return sumQueryDims;}//this variable record the total number of dimensions of all queries. It is used to indicate the number of kernels
	int getTotalnumOfQuery() const {return invert_list_spec_host.numOfQuery;}
	device_vector<QueryFeatureEnt>& get_d_query_feature_reference(){return d_query_feature;};
	device_vector<WindowQueryInfo*>& get_d_query_info_reference(){return d_query_info;};
	device_vector<GpuIndexDimensionEntry>& get_d_indexDimensionEntry_vec_reference(){return this->d_indexDimensionEntry_vec;}
	WindowQueryInfo* getQueryInfo(int queryInfo_id) {return h_query_info[queryInfo_id];}

private:

	void copy_QueryInfo_vec_fromHostToDevice(host_vector<WindowQueryInfo*>& hvec, device_vector<WindowQueryInfo*>& _dvec);
	void update_QueryInfo_entry_fromHostToDevice(WindowQueryInfo* host_queryInfo, WindowQueryInfo* device_queryInfo);
	void free_queryInfo_vec_onDevice(  device_vector<WindowQueryInfo*>& _dvec );
	template <class T>
	void copyVectorFromHostToDevice(  host_vector<T>& hvec, device_vector<T> & _dvec );
	void update_QueryInfo_upperAndLowerBound_fromHostToDevice(WindowQueryInfo* device_queryInfo, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist);

	void check_query_parameter();



public:

	/* -------------------------------- fields --------------------------------------*/
	// query specification on CPU side
	InvertListSpecGPU invert_list_spec_host;

	// top K results will be selected
	int topK;

	// inverted list and the index
	device_vector<InvlistEnt> d_invert_list;

	device_vector<int> d_invert_list_idx;

	// query to keyword: each query associate with (query_spec_host.totalDimension) keywords
	device_vector<WindowQueryInfo*> d_query_info;
	host_vector<WindowQueryInfo*> h_query_info;
	int sumQueryDims;//this variable record the total number of dimensions of all queries. It is used to indicate the number of kernels

	// result of query to feature result
	device_vector<QueryFeatureEnt> d_query_feature;

	// record which query is still running, a entry is 1 if it is still running, 0 means the query has stopped
	device_vector<int> d_valid_query;

	// number of rounds that the query has been processed
	int num_of_rounds;

	// indict whether current batch of query is initialized or not
	bool query_initialized;

	// the maximum number of dimensions, in SIFT case, max_number_of_dimensions = 128
	int max_number_of_dimensions;

	// the maximum value per dimension in the logical design of the index, in SIFT case, max_value_per_dimension = 256
	int max_value_per_dimension;

	float default_disfuncType;
	float default_upwardDistBound;
	float default_downwardDistBound;

	device_vector<GpuIndexDimensionEntry> d_indexDimensionEntry_vec;

	//device_vector<int> maxDomainForAllDimension;

};


// compare upper bound of Result object
struct Ubcomapre
{
	__host__ __device__ bool operator()(const Result &lhs, const Result &rhs) const
	{
		if (lhs.query < rhs.query) {
			return true;
		}
		else if (lhs.query > rhs.query) {
			return false;
		}

		// the query ids are the same
		if ( lhs.ub == rhs.ub ) {
			return lhs.lb < rhs.lb;
		}
		else {
			return lhs.ub < rhs.ub;
		}
	}
};

// compare lower bound of Result object
struct Lbcompare
{
	__host__ __device__ bool operator()(const Result &lhs, const Result &rhs) const
	{
		if(lhs.query < rhs.query) {
			return true;
		}
		else if(lhs.query > rhs.query) {
			return false;
		}
		return lhs.lb < rhs.lb;
	}
};



struct PartUbcompare
{
	Result rhs;

	__host__ __device__
		PartUbcompare(Result& _pivot){rhs = _pivot;}

	__host__ __device__
		bool operator()(const Result &lhs)
	{
		if(lhs.ub < rhs.ub) return true;
		else if(lhs.ub > rhs.ub) return false;
		else if(lhs.lb < rhs.lb) return true;
		else if(lhs.lb > rhs.lb) return false;
		return lhs.feature_id < rhs.feature_id;
	}
};

struct PartLbcompare
{
	Result rhs;

	__host__ __device__
		PartLbcompare(Result& _pivot){rhs = _pivot;}

	__host__ __device__
		bool operator()(const Result &lhs)
	{
		if(lhs.lb < rhs.lb) return true;
		else if(lhs.lb > rhs.lb) return false;
		else if(lhs.ub < rhs.ub) return true;
		else if(lhs.ub > rhs.ub) return false;
		return lhs.feature_id < rhs.feature_id;
	}
};



struct gCount
  {
	int thresh;

	__host__ __device__
	gCount(int& ct){ thresh = ct;}


    __host__ __device__
    bool operator()(const QueryFeatureEnt &x)
    {
      return x.count > thresh;
    }
 };


struct countFeature
{
   __host__ __device__
   QueryFeatureEnt operator()(const QueryFeatureEnt& cf, const QueryFeatureEnt& entry) const {

	   QueryFeatureEnt qf;

	   qf.count = cf.count+(entry.count);

	   return  qf;
   }
}; // end plus



#endif /* GPUMANAGER_H_ */
