/*
 * UtilityGPU.h
 *
 *  Created on: Mar 26, 2014
 *      Author: zhoujingbo
 */

#ifndef UTILITYGPU_H_
#define UTILITYGPU_H_

#include <vector>
#include <iostream>
#include <cmath>


#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;


struct GpuIndexDimensionEntry{
	int minDomain;// inclusive bound
	int maxDomain;// inclusive bound
	float bucketWidth;

	__host__ __device__ GpuIndexDimensionEntry(){
		minDomain = 0;
		maxDomain = 128;
		bucketWidth = 1;
	}
};

class GpuQueryDimensionEntry{
public:
	int dimension;
	float weight;
	float distanceFunctionPerDimension;//L-normal distance, l1, l2, or...
};


class GpuWindowQuery{

public:

	GpuWindowQuery(int queryId, int bladeId, int nds);
	GpuWindowQuery( const GpuWindowQuery& other );
	GpuWindowQuery& operator= (GpuWindowQuery other);


	// each query has the following properties
	int queryId;
	int bladeId;
	int numOfDimensionToSearch;
	int aggregateFuncType;//L-normal distance, l1, l2, or...
	std::vector<float> keywords;
	//the threshold for distance, within this threshold is considered as zero
	//use close interval
	std::vector<float> upwardDistanceBound;
	std::vector<float> downwardDistanceBound;


	//search within this bound,they are all reletive position according to query keyword
	//use close interval
	std::vector<int> depressed_upwardSearchBound;
	std::vector<int> depressed_downwardSearchBound;


	std::vector<GpuQueryDimensionEntry> depressed_dimensionSet;

	int depressed_topK;



public:

	int getNumOfDimensionToSearch() const // do not change members' values
	{
		return numOfDimensionToSearch;
	}
	void print() {

		printf("print GpuQuery=================================================================\n");
		for (int i = 0; i < numOfDimensionToSearch; i++) {
			printf(" dimensionSet[%d].dimension=%d   ", i, depressed_dimensionSet[i].dimension);
		}
		printf("\n");
		for(int i=0;i<numOfDimensionToSearch;i++){
			printf(" keywords[%d] = %f",i,keywords[i]);
		}
		printf("\n");

		for(int i=0;i<this->numOfDimensionToSearch;i++){
			printf(" upwardDistanceBound[%d] =%f ",i,upwardDistanceBound[i]);
		}
		printf("\n");

		for(int i=0;i<this->numOfDimensionToSearch;i++){
			printf(" downwardDistanceBound[%d] =%f ",i,downwardDistanceBound[i]);
		}
		printf("\n");

		for(int i=0;i<this->numOfDimensionToSearch;i++){
					printf(" dimensionSet[%d].weight =%f ",i,depressed_dimensionSet[i].weight);
		}
		printf("\n");


		printf("\n end print GpuQuery==============================================================\n");

	}

	void setDefaultDistType(float p){
		for(int i=0;i<numOfDimensionToSearch;i++){
			depressed_dimensionSet[i].distanceFunctionPerDimension = p;
		}
	}

	void depressed_setDefaultSearchBound(GpuIndexDimensionEntry* indexDimEntry){
		for(int i=0;i<numOfDimensionToSearch;i++){
			depressed_upwardSearchBound[i] = (int)std::abs(indexDimEntry[i].maxDomain-keywords[i]);
			depressed_downwardSearchBound[i] = (int)std::abs(indexDimEntry[i].minDomain-keywords[i]);
		}

	}

	//use absolute value
	void setDefaultDistanceBound(float up, float down){
		for(int i=0;i<numOfDimensionToSearch;i++){
			upwardDistanceBound[i] = up;
			downwardDistanceBound[i]=down;
		}
	}

	void setDefaultPara(int queryId, int bladeId, int nds);
};


class GPUSpecification
{
public:
	std::string appType;
	std::string searchPhase;
	std::string invertedListPath;
	std::string invertedIndexLengthFile;
	std::string searchMethod;
	int numOfDocToExpand;
	int totalDimension;
	std::vector<GpuIndexDimensionEntry> indexDimensionEntry;
	//std::vector<int> maxDomainForAllDimension;
	//std::vector<int> minDomainForAllDimension;

	float default_disfuncType;
	float default_upwardDistBound;
	float default_downwardDistBound;


public:

	void set_DefaultDimensionAndDomain(int totalDimension,
			int minDomainForAllDimension, int maxDomainForAllDimension, float bucketWidth) {
		this->totalDimension = totalDimension;
		this->indexDimensionEntry.resize(totalDimension);

		for (int i = 0; i < totalDimension; i++) {

			this->indexDimensionEntry[i].minDomain = minDomainForAllDimension; // dangerous:!!
			this->indexDimensionEntry[i].maxDomain = maxDomainForAllDimension; // dangerous:!!
			this->indexDimensionEntry[i].bucketWidth = bucketWidth;

		}
	}

	void set_query_DefaultDisFuncType(float p) {default_disfuncType=p; };
	void set_query_DefaultDistBound(float up, float down){default_upwardDistBound = up; default_downwardDistBound = down;}
};


//#define MAX_DIM 128			// max dimension size of keyword
// #define MAX_DIM_VALUE 128	// max dimension value of keyword

/* Result that returned to CPU for topK condidate */
struct Result{
	int query;		// query ID
	int feature_id;	// feature ID
	int count;	// accumulate distance
	float lb;			// lower bound
	float ub;			// upper bound

	__device__ __host__	void print_result_entry()
	{
		printf("%3d %9d %3d %5.3f %5.3f", this->query, this->feature_id, this->count, this->lb, this->ub);
	}
};

/**
 * TODO:
 * record the information for sliding window query
 */
class WindowQueryInfo
{

public:

	//__host__ __device__ QueryInfo();

	__host__ __device__ WindowQueryInfo( int numOfDimensionToSearch,int bladeId );
	__host__ __device__ WindowQueryInfo( int nds, int bladeId, float* kw, float* ubd,float* lowerBoundDist);
	__host__ __device__ WindowQueryInfo( int nds, int bladeId, float* kw);
	__host__ __device__ WindowQueryInfo( const WindowQueryInfo& other );
	__host__ WindowQueryInfo( const GpuWindowQuery& query );
	__host__ __device__ WindowQueryInfo& operator= (WindowQueryInfo other);


	__host__ __device__ ~WindowQueryInfo();


	__host__ __device__ bool checkInitialization();
	__host__ __device__ void print();
	__host__ __device__ void reset(const WindowQueryInfo& other);

private:
	__host__ __device__ void initMemberArrays(int numOfDimensionToSearch);

public:

	int depressed_topK; 	// number of results to be returned
	int numOfDimensionToSearch; // number of dimension to search
	float depressed_distFuncType;			// aggregation function
	int blade_id;

	float *keyword;		// keyword (value of each dimension) for this query
	//use  close interval
	float *upperBoundDist;		// upper bounding function to compute distance function
	float *lowerBoundDist;		// lower bounding function to compute distance function


	int2 *depressed_lastPos;		// lastPos[i].x means the down moved step, lastPos[i].y means the up moved step, all elements are valid
						// before total dimension in case the query change the search dimension in mid of search

	// all the array below, the entries are only valid up to index smaller than numOfDimensionToSearch
	int *depressed_searchDim;		// index represent the dimension to search
	float *depressed_dimWeight;	// weight of this perticular dimension
	float *depressed_distanceFunc;	// distance function for specific dimension, define the L-p normal



	// they are all reletive position according to query keyword
	//comment by jingbo: refer to upperBoundDist, change this to be float
	//these bound use the absolute value from the query point
	//use close interval, this bound is to define the search buckets
	int *depressed_upperBoundSearch;	// search bound when going up
	int *depressed_lowerBoundSearch;	// search boudn when going down
};




/* invert list specifications on GPU, won't change throughout the life time of the program*/
class InvertListSpecGPU
{
public:
	InvertListSpecGPU(int);
	InvertListSpecGPU();
	~InvertListSpecGPU();

public:
	void init_InvertListSpecGPU(int numberOfDimensions);


	int totalDimension;
	int numOfDocToExpand;
	int numOfQuery;
	GpuIndexDimensionEntry *indexDimensionEntry;	// inclusive bound of minimum and maximum domainm, bucket width
	//int *maxDomainForAllDimension;	// inclusive bound
	//int *minDomainForAllDimension;	// inclusive bound
	int maxFeatureNumber;				// exclusive number, all id must be smaller than this number
};





#endif /* UTILITYGPU_H_ */
