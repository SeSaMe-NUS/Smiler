#include <climits>
#include <limits>
#include "UtlGPU.h"


//__host__ __device__ QueryInfo::QueryInfo( )
//{
//	// apply a for the sift case
//	new (this)QueryInfo(128);
//}

GpuWindowQuery::GpuWindowQuery(int queryId, int bladeId, int nds){
	setDefaultPara(queryId, bladeId, nds);
}


GpuWindowQuery::GpuWindowQuery( const GpuWindowQuery& other )
{
	queryId = other.queryId;
	bladeId = other.bladeId;
	numOfDimensionToSearch = other.numOfDimensionToSearch;
	depressed_topK = other.depressed_topK;
	aggregateFuncType = other.aggregateFuncType;

	keywords.resize(numOfDimensionToSearch);
	depressed_upwardSearchBound.resize(numOfDimensionToSearch);
	depressed_downwardSearchBound.resize(numOfDimensionToSearch);
	upwardDistanceBound.resize(numOfDimensionToSearch);
	downwardDistanceBound.resize(numOfDimensionToSearch);
	depressed_dimensionSet.resize(numOfDimensionToSearch);

	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		keywords[i] = other.keywords[i];
		depressed_dimensionSet[i].dimension = other.depressed_dimensionSet[i].dimension;
		depressed_dimensionSet[i].distanceFunctionPerDimension = other.depressed_dimensionSet[i].distanceFunctionPerDimension;
		depressed_dimensionSet[i].weight =  other.depressed_dimensionSet[i].weight;

		upwardDistanceBound[i] = other.upwardDistanceBound[i];
		downwardDistanceBound[i] = other.downwardDistanceBound[i];

		depressed_upwardSearchBound[i] = other.depressed_upwardSearchBound[i];
		depressed_downwardSearchBound[i] = other.depressed_downwardSearchBound[i];
	}
}


GpuWindowQuery& GpuWindowQuery::operator =(GpuWindowQuery other)
{
	queryId = other.queryId;
	bladeId = other.bladeId;
	numOfDimensionToSearch = other.numOfDimensionToSearch;
	depressed_topK = other.depressed_topK;
	aggregateFuncType = other.aggregateFuncType;

	keywords.resize(numOfDimensionToSearch);
	depressed_upwardSearchBound.resize(numOfDimensionToSearch);
	depressed_downwardSearchBound.resize(numOfDimensionToSearch);
	upwardDistanceBound.resize(numOfDimensionToSearch);
	downwardDistanceBound.resize(numOfDimensionToSearch);
	depressed_dimensionSet.resize(numOfDimensionToSearch);

	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		keywords[i] = other.keywords[i];
		depressed_dimensionSet[i].dimension = other.depressed_dimensionSet[i].dimension;
		depressed_dimensionSet[i].distanceFunctionPerDimension = other.depressed_dimensionSet[i].distanceFunctionPerDimension;

		upwardDistanceBound[i] = other.upwardDistanceBound[i];
		downwardDistanceBound[i] = other.downwardDistanceBound[i];

		depressed_upwardSearchBound[i] = other.depressed_upwardSearchBound[i];
		depressed_downwardSearchBound[i] = other.depressed_downwardSearchBound[i];
	}

	return *this;
}

void GpuWindowQuery::setDefaultPara(int queryId, int bladeId, int nds){

	this->queryId = queryId;
	this->bladeId = bladeId;
	this->numOfDimensionToSearch = nds;

	depressed_topK = 5;
	aggregateFuncType = 2;

	keywords.resize(nds, 0);
	depressed_upwardSearchBound.resize(nds, INT_MAX);
	depressed_downwardSearchBound.resize(nds, INT_MAX);
	upwardDistanceBound.resize(nds, 0);
	downwardDistanceBound.resize(nds, 0);
	depressed_dimensionSet.resize(nds);

	//default setting for one GPU query
	for (int i = 0; i < numOfDimensionToSearch; i++) {

		depressed_dimensionSet[i].dimension = i;
		depressed_dimensionSet[i].distanceFunctionPerDimension = 2; //L-p normal distance, the default distance is L2
		depressed_dimensionSet[i].weight = 1;
	}

}



__host__ __device__ WindowQueryInfo::WindowQueryInfo( int numOfDimensionToSearch,int bladeId )
{
	this->numOfDimensionToSearch = numOfDimensionToSearch;
	this->blade_id= bladeId;
	//depressed_distFuncType = 2;
	//depressed_topK = 10;

	// initialization
	//initMemberArrays(numOfDimensionToSearch);
	keyword=NULL;
	upperBoundDist=NULL;
	lowerBoundDist=NULL;

	initMemberArrays(numOfDimensionToSearch);

	//for(int i = 0; i < numOfDimensionToSearch; i++)
	//{
	//	keyword[i] = -1;	// init with invalid value
		//depressed_lastPos[i].x = 0;	// init with invalid value
		//depressed_lastPos[i].y = 0;	// init with invalid value

		//depressed_searchDim[i] = -1;	// init with invalid value
		//depressed_dimWeight[i] = 1;	// init with equal weight
		//depressed_distanceFunc[i] = 2;// init with default distance function

	//	upperBoundDist[i] = 0;
	//	lowerBoundDist[i] = 0;

		// init with maximal possible value, they are all reletive position according to query keyword
		//depressed_upperBoundSearch[i] = INT_MAX;
		//depressed_lowerBoundSearch[i] = INT_MAX;
	//}
}




__host__ __device__ WindowQueryInfo::WindowQueryInfo( const WindowQueryInfo& other )
{
	numOfDimensionToSearch = other.numOfDimensionToSearch;
	blade_id=other.blade_id;

	//this->depressed_topK = other.depressed_topK;

	//this->depressed_distFuncType = other.depressed_distFuncType;

	// initialization
	initMemberArrays(numOfDimensionToSearch);

	// copy keywords
	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		// copy keywords
		this->keyword[i] = other.keyword[i];

		//depressed_lastPos[i].x = other.depressed_lastPos[i].x;
		//depressed_lastPos[i].y = other.depressed_lastPos[i].y;

		//this->depressed_searchDim[i] = other.depressed_searchDim[i];
		//this->depressed_dimWeight[i] = other.depressed_dimWeight[i];
		//this->depressed_distanceFunc[i] =	other.depressed_distanceFunc[i];


		this->upperBoundDist[i] = other.upperBoundDist[i];
		this->lowerBoundDist[i] = other.lowerBoundDist[i];


		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
		//this->depressed_upperBoundSearch[i] = other.depressed_upperBoundSearch[i];
		//this->depressed_lowerBoundSearch[i] = other.depressed_lowerBoundSearch[i];
	}

}


__host__ __device__ WindowQueryInfo::WindowQueryInfo( int nds, int bladeId, float* kw, float* ubd,float* lbd){

	this->numOfDimensionToSearch = nds;
	this->blade_id = bladeId;

	// initialization
	initMemberArrays(numOfDimensionToSearch);

	for(int i=0;i<nds;i++){
		// copy keywords
	   this->keyword[i] = kw[i];
	   this->upperBoundDist[i] = ubd[i];
	   this->lowerBoundDist[i] = lbd[i];
	}
}

__host__ __device__ WindowQueryInfo::WindowQueryInfo( int nds, int bladeId, float* kw){

	this->numOfDimensionToSearch = nds;
	this->blade_id = bladeId;
	this->keyword = new float[nds];
	for(int i=0;i<nds;i++){
			// copy keywords
		   this->keyword[i] = kw[i];
	}

	this->upperBoundDist=NULL;
	this->lowerBoundDist=NULL;
	initMemberArrays(numOfDimensionToSearch);
}

__host__ __device__ void WindowQueryInfo::reset(const WindowQueryInfo& other){
		numOfDimensionToSearch = other.numOfDimensionToSearch;
		this->blade_id = other.blade_id;

		//this->depressed_topK = other.depressed_topK;

		//this->depressed_distFuncType = other.depressed_distFuncType;


		// copy keywords
		for (int i = 0; i < numOfDimensionToSearch; i++)
		{
			// copy keywords
			this->keyword[i] = other.keyword[i];

			//depressed_lastPos[i].x = other.depressed_lastPos[i].x;
			//depressed_lastPos[i].y = other.depressed_lastPos[i].y;

			//this->depressed_searchDim[i] = other.depressed_searchDim[i];
			//this->depressed_dimWeight[i] = other.depressed_dimWeight[i];
			//this->depressed_distanceFunc[i] =	other.depressed_distanceFunc[i];


			this->upperBoundDist[i] = other.upperBoundDist[i];
			this->lowerBoundDist[i] = other.lowerBoundDist[i];


			// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
			//this->depressed_upperBoundSearch[i] = other.depressed_upperBoundSearch[i];
			//this->depressed_lowerBoundSearch[i] = other.depressed_lowerBoundSearch[i];
		}
}


__host__ __device__ WindowQueryInfo& WindowQueryInfo::operator= (WindowQueryInfo other)
{
	if ( numOfDimensionToSearch != other.numOfDimensionToSearch )
	{
		if(keyword!=NULL){
			delete[] keyword;
		}
		//delete[] depressed_lastPos;

		//delete[] depressed_searchDim;
		//delete[] depressed_dimWeight;
		//delete[] depressed_distanceFunc;

		if(upperBoundDist!=NULL){
			delete[] upperBoundDist;
		}
		if(lowerBoundDist!=NULL){
			delete[] lowerBoundDist;
		}

		//delete[] depressed_upperBoundSearch;
		//delete[] depressed_lowerBoundSearch;

		// initialization
		initMemberArrays(numOfDimensionToSearch);
	}

	numOfDimensionToSearch = other.numOfDimensionToSearch;
	this->blade_id=other.blade_id;
	//this->depressed_topK = other.depressed_topK;
	//this->depressed_distFuncType = other.depressed_distFuncType;

	// copy keywords
	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		// copy keywords
		this->keyword[i] = other.keyword[i];

		//depressed_lastPos[i].x = other.depressed_lastPos[i].x;
		//depressed_lastPos[i].y = other.depressed_lastPos[i].y;

		//this->depressed_searchDim[i] = other.depressed_searchDim[i];
		//this->depressed_dimWeight[i] = other.depressed_dimWeight[i];
		//this->depressed_distanceFunc[i] =	other.depressed_distanceFunc[i];


		this->upperBoundDist[i] = other.upperBoundDist[i];
		this->lowerBoundDist[i] = other.lowerBoundDist[i];


		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
		//this->depressed_upperBoundSearch[i] = other.depressed_upperBoundSearch[i];
		//this->depressed_lowerBoundSearch[i] = other.depressed_lowerBoundSearch[i];
	}

	return *this;
}


__host__ WindowQueryInfo::WindowQueryInfo( const GpuWindowQuery& query )
{
	this->numOfDimensionToSearch = query.getNumOfDimensionToSearch();
	this->blade_id=query.bladeId;
	//this->depressed_distFuncType = query.aggregateFuncType;

	//this->depressed_topK = query.depressed_topK;


	// initialization
	initMemberArrays(numOfDimensionToSearch);

	// copy keywords
	for (int i = 0; i < numOfDimensionToSearch; i++)
	{
		// copy keywords
		this->keyword[i] = query.keywords[i];
		this->upperBoundDist[i] = query.upwardDistanceBound[i];
		this->lowerBoundDist[i] = query.downwardDistanceBound[i];

		//depressed_lastPos[i].x = 0;	// init with invalid value
		//depressed_lastPos[i].y = 0;	// init with invalid value

		//this->depressed_searchDim[i] = query.depressed_dimensionSet[i].dimension;
		//this->depressed_dimWeight[i] = query.depressed_dimensionSet[i].weight;
		//this->depressed_distanceFunc[i] = query.depressed_dimensionSet[i].distanceFunctionPerDimension;


		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
		//this->depressed_upperBoundSearch[i] = query.depressed_upwardSearchBound[i];
		//this->depressed_lowerBoundSearch[i] = query.depressed_downwardSearchBound[i];
	}
}



/**
 * delete all the dynamic allocated arrays
 */
__host__ __device__ WindowQueryInfo::~WindowQueryInfo()
{
	if(keyword!=NULL){
		delete[] keyword;
		keyword=NULL;
	}
	//delete[] depressed_lastPos;
	//delete[] depressed_searchDim;
	//delete[] depressed_dimWeight;
	//delete[] depressed_distanceFunc;
	if(upperBoundDist!=NULL){
		delete[] upperBoundDist;
		upperBoundDist=NULL;
	}
	if(lowerBoundDist!=NULL){
		delete[] lowerBoundDist;
		lowerBoundDist=NULL;
	}
	//delete[] depressed_upperBoundSearch;
	//delete[] depressed_lowerBoundSearch;
}

__host__ __device__ void WindowQueryInfo::initMemberArrays(int nds) {

	keyword = new float[nds];
	upperBoundDist = new float[nds];
	lowerBoundDist = new float[nds];

	//depressed_lastPos = new int2[nds];

	//depressed_searchDim = new int[nds];
	//depressed_dimWeight = new float[nds];
	//depressed_distanceFunc = new float[nds];

	//depressed_upperBoundSearch = new int[nds];
	//depressed_lowerBoundSearch = new int[nds];

}

/**
 * TODO: need to be further checked
 */
__host__ __device__ bool WindowQueryInfo::checkInitialization()
{
	for(int i = 0; i < numOfDimensionToSearch; i++)
	{
		if ( keyword[i] == -1 || //depressed_searchDim[i] == -1 ||
				//depressed_upperBoundSearch[i] == INT_MAX || depressed_upperBoundSearch[i] <=0||
				//depressed_lowerBoundSearch[i] == INT_MAX || depressed_lowerBoundSearch[i] <=0||
				//depressed_dimWeight[i]<0||
				upperBoundDist[i]<0||lowerBoundDist[i]<0) {

			printf ("%s \n", "QueryInfo initialization error: The query is wrongly initialized, please check QueryInfo");
			return false;
		}
		if(lowerBoundDist[i]>upperBoundDist[i]){
			printf ("%s \n", "QueryInfo initialization error: Please make sure the min distance function bound is lower and max distance function is higher");
			return false;
		}


	}

	return true;
}


/**
 * int topK; 	// number of results to be returned
	int numOfDimensionToSearch; // number of dimension to search
	int aggregateFunc;			// aggregation function

	float *keyword;		// keyword (value of each dimension) for this query
	int2 *lastPos;		// lastPos[i].x means the down moved step, lastPos[i].y means the up moved step, all elements are valid
						// before total dimension in case the query change the search dimension in mid of search

	// all the array below, the entries are only valid up to index smaller than numOfDimensionToSearch
	int *searchDim;		// index represent the dimension to search
	float *dimWeight;	// weight of this perticular dimension
	int *distanceFunc;	// distance function for specific dimension ???

	float *upperBoundDist;		// upper bounding function to compute distance function
	float *lowerBoundDist;		// lower bounding function to compute distance function

	int *upperBoundSearch;	// search bound when going up
	int *lowerBoundSearch;	// search boudn when going down
 */


__host__ __device__ void WindowQueryInfo::print()
{

	printf("Begin QueryInfo::===================================================== \n");
	printf("numOfDimensionToSearch: %d \n", numOfDimensionToSearch);
	//printf("topK: %d \n", depressed_topK);
	//printf("aggregateFunc: %d \n", depressed_distFuncType);

	//	float *keyword;		// keyword (value of each dimension) for this query
	//int2 *lastPos;		// lastPos[i].x means the down moved step, lastPos[i].y means the up moved step, all elements are valid
						// before total dimension in case the query change the search dimension in mid of search
	printf("print keyword+++++++++++++++++++++++++++++++++++++\n");
	for(int i=0;i<numOfDimensionToSearch;i++){
		printf(" keyword[%d] = %9.1f",i,keyword[i]);
	}
	printf("\n end print keyword++++++++++++++++++++++++++++\n");

	printf("print upperBoundDist+++++++++\n");
	for(int i=0;i<this->numOfDimensionToSearch;i++){
		printf(" upperBoundDist[%d] =%f ", i, this->upperBoundDist[i]);
	}
	printf("\n end print upperBoundDist+++++++++");

	printf("print lowerBoundDist+++++++++\n");
	for(int i=0;i<this->numOfDimensionToSearch;i++){
		printf(" lowerBoundDist[%d]=%f ",i,this->lowerBoundDist[i]);
	}


	//printf("print lastPos+++++++++++++++++++++++++++++++++++\n");
	//for(int i=0;i<numOfDimensionToSearch;i++){
	//	printf(" lastPos[%d] x =%d y =%d", i, depressed_lastPos[i].x,depressed_lastPos[i].y);
	//}
	//printf("\n end print lastPos++++++++++++++++++++++++++++\n");



	//printf("print *searchDim++++++\n");
	//for (int i = 0; i < numOfDimensionToSearch; i++) {
	//	printf("searchDim[%d]=%d   ", i, depressed_searchDim[i]);
	//}
	//printf("\n");

	//printf("print *dimWeight++++++\n");
	//for (int i = 0; i < numOfDimensionToSearch; i++) {
	//	printf("dimWeight[%d]=%f   ", i, depressed_dimWeight[i]);
	//}
	//printf("\n");

	//printf("print *upperBoundSearch++++++\n");
	//for (int i = 0; i < numOfDimensionToSearch; i++) {
	//	printf("upperBoundSearch[%d]=%d   ", i, depressed_upperBoundSearch[i]);
	//}
	//printf("\n");

	//printf("print *lowerBoundSearch++++++\n");
	//for (int i = 0; i < numOfDimensionToSearch; i++) {
	//	printf("lowerBoundSearch[%d]=%d   ", i, depressed_lowerBoundSearch[i]);
	//}
	//printf("\n");

	printf(
			"\n end QueryInfo::=======================================================================\n");

//	// initialization
//	keyword = new float[numOfDimensionToSearch];
//	lastPos = new int2[numOfDimensionToSearch];
//
//	searchDim = new int[numOfDimensionToSearch];
//	dimWeight = new float[numOfDimensionToSearch];
//	distanceFunc = new int[numOfDimensionToSearch];
//
//	upperBoundDist = new float[numOfDimensionToSearch];
//	lowerBoundDist = new float[numOfDimensionToSearch];
//
//	upperBoundSearch = new int[numOfDimensionToSearch];
//	lowerBoundSearch = new int[numOfDimensionToSearch];
//
//	// copy keywords
//	for (int i = 0; i < numOfDimensionToSearch; i++)
//	{
//		// copy keywords
//		this->keyword[i] = query.keywords[i];
//
//		lastPos[i].x = 0;	// init with invalid value
//		lastPos[i].y = 0;	// init with invalid value
//
//		this->searchDim[i] = query.dimensionSet[i].dimension;
//		this->dimWeight[i] = query.dimensionSet[i].weight;
//		this->distanceFunc[i] =	query.dimensionSet[i].distanceFunctionPerDimension;
//
//
//		this->upperBoundDist[i] = query.upwardDistanceBound[i];
//		this->lowerBoundDist[i] = query.downwardDistanceBound[i];
//
//
//		// NEED TO ADD LATER: search lower bound and upper bound of different dimension.
//		this->upperBoundSearch[i] = query.upwardSearchBound[i];
//		this->lowerBoundSearch[i] = query.downwardSearchBound[i];
//	}


}


InvertListSpecGPU::InvertListSpecGPU()
{
	totalDimension = 0;

	indexDimensionEntry = NULL;

	numOfQuery = 0;
	maxFeatureNumber = 0;
	numOfDocToExpand= 0;
}


InvertListSpecGPU::InvertListSpecGPU(int numberOfDimensions)
{
	totalDimension = numberOfDimensions;

	indexDimensionEntry = new GpuIndexDimensionEntry[totalDimension];

	numOfQuery = 0;
	maxFeatureNumber = 0;
	numOfDocToExpand= 0;
}


void InvertListSpecGPU::init_InvertListSpecGPU(int numberOfDimensions)
{
	totalDimension = numberOfDimensions;

	if(indexDimensionEntry!=NULL) delete []indexDimensionEntry;
	indexDimensionEntry = new GpuIndexDimensionEntry[totalDimension];

	numOfQuery = 0;
	maxFeatureNumber = 0;
	numOfDocToExpand= 0;
}



InvertListSpecGPU::~InvertListSpecGPU()
{
	delete[] indexDimensionEntry;
}

