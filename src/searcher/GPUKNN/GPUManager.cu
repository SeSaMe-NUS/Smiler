/*
 * GPUManager.cpp
 *
 *  Created on: Apr 7, 2014
 *      Author: yuxin
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>

#include <cuda_profiler_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scan.h>



#include "GPUManager.h"
#include "generalization.h"

using namespace std;
using namespace thrust;



GPUManager::GPUManager():invert_list_spec_host(0){

	init_parameters_default();

}

GPUManager::GPUManager( GPUSpecification& gpu_spec )
{
	conf_GPUManager_GPUSpecification(gpu_spec);
}


/**
 * GPUSpecification: specify the configuration for GPUManager, i.e. index file path etc.
 * rand_invert_list_size: if inverted index file is empty, generate the data inverted index by random
 */
GPUManager::GPUManager( GPUSpecification& query_spec, int rand_invert_list_size ):
	invert_list_spec_host( query_spec.totalDimension )
{
	srand(1);

	invert_list_spec_host.numOfDocToExpand = query_spec.numOfDocToExpand;

	if ( invert_list_spec_host.numOfDocToExpand <= 0 )
	{
		cout << "cannot handle numOfDocToExpand is equal or below 0" << endl;
		exit(1);
	}

	for ( int i = 0; i < invert_list_spec_host.totalDimension; i++ )
	{
		invert_list_spec_host.indexDimensionEntry[i] = query_spec.indexDimensionEntry[i];

	}

	this->set_query_DefaultDisFuncType(query_spec.default_disfuncType);
	this->set_query_DefaultDistBound(query_spec.default_upwardDistBound,query_spec.default_downwardDistBound);

	// read inverted list
	if ( query_spec.invertedListPath.empty() && query_spec.invertedIndexLengthFile.empty() )
	{
		invert_list_spec_host.maxFeatureNumber = rand_invert_list_size;
		rand_inv_list();
	}
	else
	{
		// read inverted list from file
		//this->get_invert_list_from_file(query_spec.invertedListPath,query_spec.invertedIndexLengthFile);
		int numofDim=0;//out result
		int maxValuePerDim = 0;//output result
		int maxFeatureID = 0;
		get_invert_list_from_binary_file( query_spec.invertedListPath, numofDim, maxValuePerDim,maxFeatureID);//Note: also set the maxFeatureID

		this->max_number_of_dimensions = numofDim;
		this->max_value_per_dimension = maxValuePerDim;
		this->invert_list_spec_host.maxFeatureNumber = maxFeatureID;
	}

	clear_query_memory();


	exec_time.resize(15,0);
}


GPUManager::~GPUManager()
{

	//freeQueryInfoDevice(d_query_info);
	clear_query_memory();

}

void GPUManager::init_parameters_default(){
		this->num_of_rounds = 0;

		this->max_number_of_dimensions = 0;
		this->max_value_per_dimension = 0;
		this->invert_list_spec_host.maxFeatureNumber = 0;
		this->query_initialized = false;
		sumQueryDims=0;
		default_disfuncType = 2;
		default_upwardDistBound =0;
		default_downwardDistBound=0;
		topK = 0;
}

void GPUManager::conf_GPUManager_GPUSpecification(GPUSpecification& gpu_spec){



		invert_list_spec_host.init_InvertListSpecGPU(gpu_spec.totalDimension );
		invert_list_spec_host.numOfDocToExpand = gpu_spec.numOfDocToExpand;

		if ( invert_list_spec_host.numOfDocToExpand <= 0 )
		{
			cout << "cannot handle numOfDocToExpand is equal or below 0" << endl;
			exit(1);
		}

		for(int i = 0; i < invert_list_spec_host.totalDimension; i++)
		{
			invert_list_spec_host.indexDimensionEntry[i] = gpu_spec.indexDimensionEntry[i];

		}

		this->set_query_DefaultDisFuncType(gpu_spec.default_disfuncType);
		this->set_query_DefaultDistBound(gpu_spec.default_upwardDistBound,gpu_spec.default_downwardDistBound);

		// read inverted list from file
		int numofDim=0;//out result
		int maxValuePerDim = 0;//output result
		int maxFeatureNumber = 0;

		//this function also set the invert_list_spec_host.maxFeatureID which determinese the size of count&ACD table
		get_invert_list_from_binary_file( gpu_spec.invertedListPath, numofDim, maxValuePerDim,maxFeatureNumber);

		this->max_number_of_dimensions = numofDim;
		this->max_value_per_dimension = maxValuePerDim;
		this->invert_list_spec_host.maxFeatureNumber = maxFeatureNumber;

		clear_query_memory();


		exec_time.resize(15,0);
}

void GPUManager::clear_query_memory(){

	free_queryInfo_vec_onDevice(d_query_info);
	d_query_info.clear();
	d_query_info.shrink_to_fit();

	for(int i=0;i<this->h_query_info.size();i++){
		delete h_query_info[i];
	}
	h_query_info.clear();


	sumQueryDims=0;

	d_query_feature.clear();
	d_query_feature.shrink_to_fit();

	d_valid_query.clear();
	d_valid_query.shrink_to_fit();
	this->query_initialized = false;


}



/**
 *
 * TODO:
 *
 *
 */
void GPUManager::init_GPU_query( vector<GpuWindowQuery>& query_set )
{
	clear_query_memory();

	if( query_set.empty() )
	{
		exit(1);
	}
	else
	{

		this->invert_list_spec_host.numOfQuery = query_set.size();
		//host_vector<QueryInfo*> h_query_info;


		h_query_info.reserve( this->invert_list_spec_host.numOfQuery );


		for(int i = 0; i < this->invert_list_spec_host.numOfQuery; i++)
		{
			WindowQueryInfo *queryInfo = new WindowQueryInfo( query_set[i] );
			sumQueryDims += queryInfo->numOfDimensionToSearch;
			h_query_info.push_back( queryInfo );
		}


		d_query_info.reserve( h_query_info.size() );

		// copy queryInfo to gpu
		copy_QueryInfo_vec_fromHostToDevice( h_query_info, d_query_info );




	}
	//here!: test wether correctly initialized
	initialize_dev_constMem( invert_list_spec_host );
	printf("#total number of queries in GPUMananger %d\n",invert_list_spec_host.numOfQuery);


	d_indexDimensionEntry_vec.clear();
	d_indexDimensionEntry_vec.reserve(invert_list_spec_host.totalDimension);

	for ( int i = 0; i < invert_list_spec_host.totalDimension; i++ )
	{
		d_indexDimensionEntry_vec.push_back( invert_list_spec_host.indexDimensionEntry[i] );
	}


	// checking memory size
	if( (double) invert_list_spec_host.maxFeatureNumber * invert_list_spec_host.numOfQuery * sizeof(QueryFeatureEnt) > std::numeric_limits<int>::max() )
	{
		cout << "too much memory is used for query feature mapping" << endl;
		exit(1);
	}

	//d_query_feature = new device_vector<QueryFeatureEnt>( (unsigned int) invert_list_spec_host.maxFeatureID * invert_list_spec_host.numOfQuery );
	d_query_feature.clear();
	d_query_feature.resize((unsigned int) invert_list_spec_host.maxFeatureNumber * invert_list_spec_host.numOfQuery);
	d_query_feature.shrink_to_fit();
	//d_valid_query = new device_vector<int>( invert_list_spec_host.numOfQuery, 1 );
	d_valid_query.clear();
	d_valid_query.resize(invert_list_spec_host.numOfQuery,1);
	d_valid_query.shrink_to_fit();
	num_of_rounds = 0;

	this->query_initialized = true;
	cout << "finish initialization" << endl;
}



struct gzero
  {
    __host__ __device__
    bool operator()(const QueryFeatureEnt &x)
    {
      return x.count > 0;
    }
 };



bool GPUManager::bi_direction_query_KernelPerDim (int threshold, int topKValue, vector<Result>& _result )
{



	check_query_parameter();

	cudaProfilerStart();
	cudaEvent_t start, stop;
	cudaEvent_t gs, ge;
	float elapsedTime;
	float elapseTimeForCount;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&gs);
	cudaEventCreate(&ge);

	cout	<< "GPUManager::bi_direction_query(): 0 -- compute_mapping_saving_pos_KernelPerDim()"
			<< endl;
	cudaEventRecord(gs, 0);
	cudaEventRecord(start, 0);


	compute_mapping_saving_pos_KernelPerDim_template<<<sumQueryDims, THREAD_PER_BLK>>>(
			raw_pointer_cast(this->d_query_info.data()),
			raw_pointer_cast(this->d_invert_list.data()),
			raw_pointer_cast(this->d_invert_list_idx.data()),
			raw_pointer_cast(this->d_query_feature.data()),	//QueryFeatureEnt* query_feature, i.e. count&ACD table
			false, max_value_per_dimension,
			raw_pointer_cast(this->d_indexDimensionEntry_vec.data()),
			DataToIndex_keywordMap_bucketUnit(),
			IndexToData_lastPosMap_bucketUnit(),
			Lp_distance()
			);



	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[0] += elapsedTime;

	cout	<< "GPUManager::bi_direction_query(): 1 -- prefix_count_KernelPerQuery()"
			<< endl;
	device_vector<int> threshold_count(	invert_list_spec_host.numOfQuery * THREAD_PER_BLK, 0);// prefix count for each thread
	device_vector<int> query_result_count(invert_list_spec_host.numOfQuery, 0);	// prefix count for each block (i.e. each query)
	elapseTimeForCount=0;

	cudaEventRecord(start, 0);
	prefix_count_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK,THREAD_PER_BLK*sizeof(int)>>>(
			raw_pointer_cast(this->d_query_feature.data()),
			raw_pointer_cast(threshold_count.data()),
			raw_pointer_cast(query_result_count.data()),
			threshold
			);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapseTimeForCount+=elapsedTime;
	exec_time[7] += elapsedTime;

	cudaEventRecord(start, 0);
	thrust::inclusive_scan(threshold_count.begin(), threshold_count.end(),
			threshold_count.begin()); // per thread inclusive scan
	cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		elapseTimeForCount+=elapsedTime;
		exec_time[8] += elapsedTime;
	cudaEventRecord(start, 0);
	thrust::inclusive_scan(query_result_count.begin(), query_result_count.end(),
			query_result_count.begin()); // per block inclusive scan
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapseTimeForCount+=elapsedTime;
	exec_time[9] += elapsedTime;
	exec_time[1] += elapseTimeForCount;

	cout	<< "GPUManager::bi_direction_query(): 2 -- output_result_bidrection_search_KernelPerQuery()"
			<< endl;
	device_vector<Result> d_result_ub_sorted(threshold_count[threshold_count.size() - 1]);
	//cout <<" the Count sum is:"<<query_result_count[invert_list_spec_host.numOfQuery-1]<<endl;
	cudaEventRecord(start, 0);
	/*output_result_bidrection_search_KernelPerQuery<<<
			this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK,
			3 * max_number_of_dimensions*sizeof(float)>>>(
			raw_pointer_cast(this->d_query_feature.data()),
			raw_pointer_cast(this->d_query_info.data()),
			raw_pointer_cast(threshold_count.data()),
			raw_pointer_cast(d_result_ub_sorted.data()), //record the lower bound and upper bound
			threshold, raw_pointer_cast(this->d_vec_indexDimensionEntry.data())
			//raw_pointer_cast(this->maxDomainForAllDimension.data())//remove
			);*/

	output_result_bidrection_search_KernelPerQuery_template<<<
				this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK,
				3 * max_number_of_dimensions*sizeof(float)>>>(
				raw_pointer_cast(this->d_query_feature.data()),
				raw_pointer_cast(this->d_query_info.data()),
				raw_pointer_cast(threshold_count.data()),
				raw_pointer_cast(d_result_ub_sorted.data()), //record the lower bound and upper bound
				threshold, raw_pointer_cast(this->d_indexDimensionEntry_vec.data()),
				//raw_pointer_cast(this->maxDomainForAllDimension.data())//remove
				DataToIndex_keywordMap_bucketUnit(),
				IndexToData_lastPosMap_bucketUnit(),
				Lp_distance()
				);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[2] += elapsedTime;

	cout << "GPUManager::bi_direction_query(): 3 -- sort d_result_lb & ub"
			<< endl;
	//device_vector<Result> d_result_temp(d_result_ub_sorted.size());
	device_vector<Result> d_result_lb_sorted(d_result_ub_sorted.begin(),
			d_result_ub_sorted.end());
	cudaEventRecord(start, 0);
	//thrust::sort(d_result_ub_sorted.begin(), d_result_ub_sorted.end(),
	//		Ubcomapre());
	//thrust::sort(d_result_lb_sorted.begin(), d_result_lb_sorted.end(),
	//		Lbcompare());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[3] += elapsedTime;

	cout
			<< "GPUManager::bi_direction_query(): 4 -- terminate_check_KernelPerQuery()"
			<< endl;
	device_vector<Result> d_result_temp(d_result_ub_sorted.size());//create temp buffer for checking
	cudaEventRecord(start, 0);


	terminateCheck_kSelection_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,	THREAD_PER_BLK>>>(
			raw_pointer_cast(d_result_ub_sorted.data()),
			//raw_pointer_cast(d_result_lb_sorted.data()),
			raw_pointer_cast(d_result_temp.data()),
			raw_pointer_cast(query_result_count.data()),
			raw_pointer_cast(d_valid_query.data()), topKValue
			);
	int terminate_sum = thrust::reduce(d_valid_query.begin(),
			d_valid_query.end());


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[4] += elapsedTime;


	cout<< "GPUManager::bi_direction_query(): 4 -- extract_topK_KernelPerQuery()"<< endl;
	device_vector<Result> d_result(	this->invert_list_spec_host.numOfQuery * topKValue);
	cudaEventRecord(start, 0);
	extract_topK_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK>>>(
			raw_pointer_cast(d_result_ub_sorted.data()),
			raw_pointer_cast(query_result_count.data()),
			raw_pointer_cast(d_result.data()), topKValue);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[5] += elapsedTime;

	cudaEventRecord(ge, 0);
	cudaEventSynchronize(ge);
	cudaEventElapsedTime(&elapsedTime, gs, ge);
	exec_time[6] += elapsedTime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaProfilerStop();

	num_of_rounds++;

	_result.resize(d_result.size());
	HANDLE_ERROR(
			cudaMemcpy(&_result[0], raw_pointer_cast(d_result.data()),
					d_result.size() * sizeof(Result), cudaMemcpyDeviceToHost));



	// no more query should be made to this batch
	if (terminate_sum == 0) {
		this->query_initialized = false;
	}

	return terminate_sum == 0;
}






//==============================================================================
void GPUManager::point_query(vector<Result>& result,int threshold){

	check_query_parameter();

	compute_mapping_saving_pos_KernelPerDim_template<<< this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK, max_number_of_dimensions >>>(
		raw_pointer_cast(this->d_query_info.data()),
		raw_pointer_cast(this->d_invert_list.data()),
		raw_pointer_cast(this->d_invert_list_idx.data()),
		raw_pointer_cast(this->d_query_feature.data()),
		true,
		max_value_per_dimension,
		raw_pointer_cast( this->d_indexDimensionEntry_vec.data() ),
		DataToIndex_keywordMap_bucketUnit(),
		IndexToData_lastPosMap_bucketUnit(),
		Lp_distance()
		);

	device_vector<int> threshold_count(invert_list_spec_host.numOfQuery*THREAD_PER_BLK,0);	// prefix count for each thread
	device_vector<int> query_result_count(invert_list_spec_host.numOfQuery,0);				// prefix count for each block (query)

	prefix_count_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK>>>(
		raw_pointer_cast(this->d_query_feature.data()),
		raw_pointer_cast(threshold_count.data()),
		raw_pointer_cast(query_result_count.data()),
		threshold
		);

	thrust::inclusive_scan(threshold_count.begin(),threshold_count.end(),threshold_count.begin()); // per thread inclusive scan
	thrust::inclusive_scan(query_result_count.begin(),query_result_count.end(),query_result_count.begin()); // per block inclusive scan

	device_vector<Result> d_result(threshold_count[threshold_count.size()-1]);

	output_result_point_search<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK>>>(
		raw_pointer_cast(this->d_query_feature.data()),
		raw_pointer_cast(threshold_count.data()),
		raw_pointer_cast(d_result.data()),
		threshold);

	result.resize(d_result.size());
	HANDLE_ERROR( cudaMemcpy(&result[0],raw_pointer_cast(d_result.data()),d_result.size() * sizeof(Result), cudaMemcpyDeviceToHost) );

	this->query_initialized = false; // make sure no other query is called
}



/**
 * retrieve invert list from binary given the filename, and read the number of dimension and max value per
 *  dimension from the binary file
 */
void GPUManager::get_invert_list_from_binary_file ( string filename, int& _numberOfDim,  int& _maxValuePerDim, int& _maxFeatureNumber )
{
		// initialize the previousKey to be 0;
		int previousKey = -1;
		int key;
		int numberOfFeatures;
		int keyPosition = 0;
		int offset = 0;
		_maxFeatureNumber = -1;

		cout << "file name is: " << filename << endl;

		vector<InvlistEnt>  featureIdList;
		vector<int>  featureIdListIndex;

		clock_t start = clock();
		ifstream inFile(filename.c_str(), ios::in | ios::binary );


		if ( ! inFile )
		{
			cerr << "error : unable to open input file 1: " << inFile << endl;
		}
		else if ( ! inFile.is_open() )
		{
			cerr << "error : unable to open input file 2: " << inFile << endl;
		}
		else
		{
			int numberOfDim;
			inFile.read((char*) (&numberOfDim), sizeof(int) );
			_numberOfDim=numberOfDim;

			int maxValuePerDim;
			inFile.read((char*) (&maxValuePerDim), sizeof(int) );
			_maxValuePerDim=maxValuePerDim;


			// assert( MAX_DIM_VALUE == maxValuePerDim );

			// read the key, if reach the end of the file, jump out
			while ( inFile.read((char*) (&key), sizeof(int) ) )
			{
				int dim_value = key % maxValuePerDim;
				int dim = key / maxValuePerDim;

				assert(dim<invert_list_spec_host.totalDimension &&("error::dim is outside of maximum possible dimensions\n"));

				/*if ( dim > invert_list_spec_host.totalDimension -1 )
				{
					cout << "dim: " << dim << endl;
					cout << "key: " << key << endl;
					cout << "read file problem" << endl;
				}*/

				assert((dim_value>=invert_list_spec_host.indexDimensionEntry[dim].minDomain &&
						dim_value<=invert_list_spec_host.indexDimensionEntry[dim].maxDomain)&&
						("error:input invert list has problem for dimension values, should be between min and max dimension value\n"));

				/*if ( dim_value < invert_list_spec_host.minDomainForAllDimension[dim] ||
					 dim_value > invert_list_spec_host.maxDomainForAllDimension[dim] )
				{
						cout << "input invert list has problem for dimension values, should be between min and max dimension value" << endl;
						cout << "dimension " << dim << " min: " << invert_list_spec_host.minDomainForAllDimension[dim] << " and max: " << invert_list_spec_host.maxDomainForAllDimension[dim] << endl;
						cout << "value is: " << dim_value << endl;
						cout << "key: " << key << endl;
						exit(1);
				}*/

				// read the number of features
				inFile.read((char*) (&numberOfFeatures), sizeof(int) );

				// read the feature ids, and fill in the featureIdList
				if ( numberOfFeatures > 0 )
				{
					int featureID;
					for ( int j = 0; j < numberOfFeatures; j++ )
					{

						// read the feature ID list
						inFile.read((char*) (&featureID), sizeof(int) );
						InvlistEnt new_ent = featureID;
						if(new_ent > _maxFeatureNumber ){
							_maxFeatureNumber  = new_ent;
						}
						featureIdList.push_back(new_ent);

					}

				}

				// get the offset between keys
				offset = key - previousKey;

				// add the position for empty keys
				for ( int j = 0; j < offset - 1; j++ )
				{
					featureIdListIndex.push_back(keyPosition);
				}

				// update the current position
				keyPosition += numberOfFeatures;
				// add the position for the current key
				featureIdListIndex.push_back(keyPosition);

				// update the previous position
				previousKey = key;
			}

			// fill the key positions for the empty keys
			int maxNumberOfList = this->invert_list_spec_host.totalDimension * _maxValuePerDim;
			offset = maxNumberOfList - key - 1;
			for ( int j = 0; j < offset; j++ )
			{
				featureIdListIndex.push_back(keyPosition);
			}

			inFile.close();
			inFile.clear();
		}

//		featureIdListIndex.erase(featureIdListIndex.begin());


		double init_time = (double)(clock() - start) / CLOCKS_PER_SEC;
		cout<< "Reading Inverted Lists from disk to Host Memory takes: " << init_time << " seconds. " <<endl;
		//invert_list_spec_host.maxFeatureID++;
		_maxFeatureNumber++;//exclusive feature Id, all smaller than this value

		start = clock();
		host_vector<InvlistEnt> featureIdList_h(featureIdList);
		host_vector<int>  featureIdListIndex_h(featureIdListIndex);

		cout << "size of list featureIdList_h: " << featureIdList_h.size()  << endl;

		cout << "size of list  featureIdList_h* sizeof(InvlistEnt): " << featureIdList_h.size() * sizeof(InvlistEnt) << endl;

		cout << "size of list featureIdListIndex_h: " << featureIdListIndex_h.size()  << endl;

		cout<< "host vector allocated" <<endl;
		//d_invert_list =new device_vector<device_vector>;
		copyVectorFromHostToDevice( featureIdList_h, d_invert_list );
		copyVectorFromHostToDevice( featureIdListIndex_h, d_invert_list_idx );

		init_time = (double)(clock() - start) / CLOCKS_PER_SEC;
		cout<< "Loading inverted index from CPU to GPU takes: " << init_time << " seconds. " <<endl;


}




/**
 * generate random inverted list
 */
void GPUManager::rand_inv_list()
{
	vector<vector<int> > keyword_feature(invert_list_spec_host.totalDimension * max_value_per_dimension);
	for(int i = 0; i < invert_list_spec_host.maxFeatureNumber; i++){
		for(int x = 0; x < invert_list_spec_host.totalDimension;  x++){
			int dim_range = invert_list_spec_host.indexDimensionEntry[x].maxDomain - invert_list_spec_host.indexDimensionEntry[x].minDomain;
			int dim_base = invert_list_spec_host.indexDimensionEntry[x].minDomain;
			int dim_value = rand() % dim_range + dim_base;
			int keyword = x * max_value_per_dimension + dim_value;
			keyword_feature[keyword].push_back(i);
		}
	}

	host_vector<InvlistEnt> h_invert_list;
	host_vector<int> h_invert_list_idx;
	int count = 0;
	// construct inverted list
	for(int i = 0; i < invert_list_spec_host.totalDimension * max_value_per_dimension; i++){
		for(int j = 0; j < keyword_feature[i].size(); j++)
		{
			InvlistEnt new_ent;
			new_ent = keyword_feature[i][j];
			h_invert_list.push_back(new_ent);
		}
		count += keyword_feature[i].size();
		h_invert_list_idx.push_back(count);
	}

	// transfer to gpu
	copyVectorFromHostToDevice( h_invert_list, d_invert_list );
	copyVectorFromHostToDevice( h_invert_list_idx, d_invert_list_idx );
}




/**
 * read query from file, the query is a list of vectors, i.e. no additional informatino is provided such as weight and bound
 */
void GPUManager::readQueryFromFile( string queryFileName, vector<GpuWindowQuery>& _query_set )
{

	if(queryFileName.empty())
	{
		return;
	}

	ifstream query_file(queryFileName.c_str());

	if(!query_file.is_open())
	{
		cout << "query file "<<queryFileName<<" cannot found" << endl;
		exit(1);
	}


	//host_vector<QueryInfo> h_query_info;
	int num_query = 0;

	while(!query_file.eof())
	{
		string input_line;
		getline( query_file, input_line );

		if( input_line.empty() )
		{
			continue;
		}

		stringstream ss(input_line);

		// initialize GpuQuery
		GpuWindowQuery gq(num_query, 0, invert_list_spec_host.totalDimension) ;//0 is not correct, but this function will be removed

		for (int j = 0; j < invert_list_spec_host.totalDimension; j++)
		{
			gq.depressed_dimensionSet[j].dimension = j;
			float dim_value = -1;
			ss >> dim_value;
			gq.keywords[j] = dim_value;
		}

		gq.setDefaultDistType(this->default_disfuncType);
		gq.depressed_setDefaultSearchBound(invert_list_spec_host.indexDimensionEntry);
		gq.setDefaultDistanceBound(this->default_upwardDistBound,this->default_downwardDistBound);
		_query_set.push_back(gq);
		num_query++;
	}

	cout<< "query set size: " << _query_set.size() <<endl;
	cout<< "readQueryFromFile(): finished loading query file" <<endl;
}






void GPUManager::print_query_doucment_mapping(){
	cout << "query document mapping" << endl;
	host_vector<QueryFeatureEnt> host_mapping = (this->d_query_feature);

	for(int i = 0; i < invert_list_spec_host.numOfQuery; i++)
	{
		for(int j = 0; j < invert_list_spec_host.maxFeatureNumber; j++)
		{
			QueryFeatureEnt entry = host_mapping[i*invert_list_spec_host.maxFeatureNumber+j];
			cout << entry.count << " ";
		}
		cout << endl;
	}
	cout << "++++++++++++++++++++++++++++++++++++++" << endl;
}


void GPUManager::print_invert_list(){
	cout << "invert list" << endl;
	host_vector<InvlistEnt> host_invert_list = (this->d_invert_list);
	host_vector<InvlistEnt> host_invert_list_idx = (this->d_invert_list_idx);
	for (int i = 0; i < host_invert_list_idx.size(); i++ )
	{
		int dim = i / max_value_per_dimension;
		int value = i % max_value_per_dimension;
		cout << "(" << dim << "," << value << "):";
		int start = i == 0 ? 0 : host_invert_list_idx[i-1];
		int end = host_invert_list_idx[i];
		for(int j = start; j < end; j++) cout << host_invert_list[j] << " ";
		cout << endl;
	}
	cout << "++++++++++++++++++++++++++++++++++++++" << endl;
}




void GPUManager::print_result_vec(vector<Result>& result_vec)
{
	cout << "result of output: " << endl;
	for ( int i = 0; i < result_vec.size(); i++)
	{
		result_vec[i].print_result_entry();
		cout << endl;
	}
	cout << "++++++++++++++++++++++++++++++++++++++" << endl;
}

//GPU open API, provide basic GPU function via these APIs
template <class KEYWORDMAP, class LASTPOSMAP, class DISTFUNC>
void GPUManager::dev_BidirectionExpansion(
		KEYWORDMAP keywordMap,
		LASTPOSMAP lastPosMap,
		DISTFUNC distFunc){

	compute_mapping_saving_pos_KernelPerDim_template<<<sumQueryDims, THREAD_PER_BLK>>>(
				raw_pointer_cast(this->d_query_info.data()),
				raw_pointer_cast(this->d_invert_list.data()),
				raw_pointer_cast(this->d_invert_list_idx.data()),
				raw_pointer_cast(this->d_query_feature.data()),	//QueryFeatureEnt* query_feature, i.e. count&ACD table
				false, max_value_per_dimension,//logical unit, caculated by the number of bits allocated for value
				raw_pointer_cast(this->d_indexDimensionEntry_vec.data()),
				keywordMap,
				lastPosMap,
				distFunc
				);
}

template void GPUManager::dev_BidirectionExpansion<DataToIndex_keywordMap_bucketUnit, IndexToData_lastPosMap_bucketUnit, Lp_distance>(
		DataToIndex_keywordMap_bucketUnit keywordMap,
		IndexToData_lastPosMap_bucketUnit lastPosMap,
		Lp_distance distFunc);

template void GPUManager::dev_BidirectionExpansion<DataToIndex_keywordMap_bucket, IndexToData_lastPosMap_bucket_exclusive, Lp_distance>(
		DataToIndex_keywordMap_bucket keywordMap,
		IndexToData_lastPosMap_bucket_exclusive lastPosMap,
		Lp_distance distFunc);

void GPUManager::update_windowQuery_entryAndTable(WindowQueryInfo* host_queryInfo, int d_query_info_id){

	update_QueryInfo_entry_fromHostToDevice( host_queryInfo, this->d_query_info[d_query_info_id]);
	delete h_query_info[ d_query_info_id];
	h_query_info[ d_query_info_id]  = host_queryInfo;

	int wqi_start = d_query_info_id*this->getMaxFeatureNumber();
	int wqi_end = wqi_start+this->getMaxFeatureNumber();

	thrust::fill(this->d_query_feature.begin()+wqi_start,this->d_query_feature.begin()+wqi_end,QueryFeatureEnt());//clear the Count&ACD table of this window query

}

void GPUManager::update_windowQuery_entryAndTable(GpuWindowQuery& windowQuery){

	WindowQueryInfo* hqi = new WindowQueryInfo(windowQuery);

	update_windowQuery_entryAndTable(hqi, windowQuery.queryId);


	//delete hqi;

}

void GPUManager::update_QueryInfo_entry_upperAndLowerBound(int queryId, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist){

	//update device
	update_QueryInfo_upperAndLowerBound_fromHostToDevice(this->d_query_info[queryId], new_upperBoundDist,  new_lowerBoundDist);
	//update host
	for(int i=0;i<this->h_query_info[queryId]->numOfDimensionToSearch;i++){
		h_query_info[queryId]->upperBoundDist[i] = new_upperBoundDist[i];
		h_query_info[queryId]->lowerBoundDist[i] = new_lowerBoundDist[i];
	}

}



//================================private function

/**
 *
 */
void GPUManager::update_QueryInfo_entry_fromHostToDevice(WindowQueryInfo* host_queryInfo, WindowQueryInfo* device_queryInfo){

		//copy data

		cudaMemcpy(&(device_queryInfo->depressed_topK), &(host_queryInfo->depressed_topK), sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&(device_queryInfo->numOfDimensionToSearch), &(host_queryInfo->numOfDimensionToSearch), sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(&(device_queryInfo->depressed_distFuncType), &(host_queryInfo->depressed_distFuncType), sizeof(int), cudaMemcpyHostToDevice);

		int size = host_queryInfo->numOfDimensionToSearch;
		WindowQueryInfo* d2h_queryInfo = (WindowQueryInfo*) malloc(sizeof(WindowQueryInfo));
		cudaMemcpy(d2h_queryInfo, device_queryInfo, sizeof(WindowQueryInfo), cudaMemcpyDeviceToHost);

		cudaMemcpy(d2h_queryInfo->depressed_searchDim, host_queryInfo->depressed_searchDim, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->depressed_distanceFunc, host_queryInfo->depressed_distanceFunc, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->depressed_upperBoundSearch, host_queryInfo->depressed_upperBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->depressed_lowerBoundSearch, host_queryInfo->depressed_lowerBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->depressed_lastPos, host_queryInfo->depressed_lastPos, sizeof(int2) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->keyword, host_queryInfo->keyword, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->depressed_dimWeight, host_queryInfo->depressed_dimWeight, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->upperBoundDist, host_queryInfo->upperBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->lowerBoundDist, host_queryInfo->lowerBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);
		free(d2h_queryInfo);

}

void GPUManager::update_QueryInfo_upperAndLowerBound_fromHostToDevice(WindowQueryInfo* device_queryInfo, vector<float>& new_upperBoundDist, vector<float>& new_lowerBoundDist){

		int size = new_upperBoundDist.size();
		WindowQueryInfo* d2h_queryInfo = (WindowQueryInfo*) malloc(sizeof(WindowQueryInfo));
		cudaMemcpy(d2h_queryInfo, device_queryInfo, sizeof(WindowQueryInfo), cudaMemcpyDeviceToHost);

		cudaMemcpy(d2h_queryInfo->upperBoundDist, new_upperBoundDist.data(), sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(d2h_queryInfo->lowerBoundDist, new_lowerBoundDist.data(), sizeof(float) * size, cudaMemcpyHostToDevice);

		free(d2h_queryInfo);
}



void GPUManager::copy_QueryInfo_vec_fromHostToDevice(  host_vector<WindowQueryInfo*>& hvec, device_vector<WindowQueryInfo*>& _dvec )
{

	for ( int i = 0; i < hvec.size(); i++ )
	{
		WindowQueryInfo* hostQueryInfo = hvec[i];
		int size = hostQueryInfo->numOfDimensionToSearch;


		// copy class to gpu
		WindowQueryInfo* devQueryInfo;
		cudaMalloc( (void **) &devQueryInfo, sizeof(WindowQueryInfo) );
		cudaMemcpy(devQueryInfo, hostQueryInfo, sizeof(WindowQueryInfo), cudaMemcpyHostToDevice);

		// copy real data to gpu
		int *searchDim;
		float *distanceFunc;
		int *upperBoundSearch;
		int *lowerBoundSearch;
		int2 *lastPos;
		float *keyword;
		float *dimWeight;
		float *upperBoundDist;
		float *lowerBoundDist;

		cudaMalloc((void **) &searchDim, sizeof(int) * size);
		cudaMalloc((void **) &distanceFunc, sizeof(int) * size);
		cudaMalloc((void **) &upperBoundSearch, sizeof(int) * size);
		cudaMalloc((void **) &lowerBoundSearch, sizeof(int) * size);
		cudaMalloc((void **) &lastPos, sizeof(int2) * size);
		cudaMalloc((void **) &keyword, sizeof(float) * size);
		cudaMalloc((void **) &dimWeight, sizeof(float) * size);
		cudaMalloc((void **) &upperBoundDist, sizeof(float) * size);
		cudaMalloc((void **) &lowerBoundDist, sizeof(float) * size);


		cudaMemcpy(searchDim, hostQueryInfo->depressed_searchDim, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(distanceFunc, hostQueryInfo->depressed_distanceFunc, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(upperBoundSearch, hostQueryInfo->depressed_upperBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lowerBoundSearch, hostQueryInfo->depressed_lowerBoundSearch, sizeof(int) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lastPos, hostQueryInfo->depressed_lastPos, sizeof(int2) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(keyword, hostQueryInfo->keyword, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(dimWeight, hostQueryInfo->depressed_dimWeight, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(upperBoundDist, hostQueryInfo->upperBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);
		cudaMemcpy(lowerBoundDist, hostQueryInfo->lowerBoundDist, sizeof(float) * size, cudaMemcpyHostToDevice);


		// copy pointer to gpu
		cudaMemcpy(&(devQueryInfo->depressed_searchDim), &searchDim, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_distanceFunc), &distanceFunc, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_upperBoundSearch), &upperBoundSearch, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_lowerBoundSearch), &lowerBoundSearch, sizeof(int *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_lastPos), &lastPos, sizeof(int2 *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->keyword), &keyword, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->depressed_dimWeight), &dimWeight, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->upperBoundDist), &upperBoundDist, sizeof(float *), cudaMemcpyHostToDevice);
		cudaMemcpy(&(devQueryInfo->lowerBoundDist), &lowerBoundDist, sizeof(float *), cudaMemcpyHostToDevice);

		_dvec.push_back(devQueryInfo);


	}

}

void GPUManager::free_queryInfo_vec_onDevice(  device_vector<WindowQueryInfo*>& _dvec ){

	for ( int i = 0; i < _dvec.size(); i++ )
		{
			// copy class reference to cpu
			WindowQueryInfo* devQueryInfo = _dvec[i];

			int *searchDim;
			cudaMemcpy(&searchDim,&(devQueryInfo->depressed_searchDim), sizeof(int *), cudaMemcpyDeviceToHost);
			cudaFree(searchDim);

			float *distanceFunc;
			cudaMemcpy(&distanceFunc,&(devQueryInfo->depressed_distanceFunc), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(distanceFunc);

			int *upperBoundSearch;
			cudaMemcpy(&upperBoundSearch,&(devQueryInfo->depressed_upperBoundSearch), sizeof(int *), cudaMemcpyDeviceToHost);
			cudaFree(upperBoundSearch);

			int *lowerBoundSearch;
			cudaMemcpy(&lowerBoundSearch,&(devQueryInfo->depressed_lowerBoundSearch), sizeof(int *), cudaMemcpyDeviceToHost);
			cudaFree(lowerBoundSearch);

			int2 *lastPos;
			cudaMemcpy(&lastPos,&(devQueryInfo->depressed_lastPos), sizeof(int2 *), cudaMemcpyDeviceToHost);
			cudaFree(lastPos);

			float *keyword;
			cudaMemcpy(&keyword,&(devQueryInfo->keyword), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(keyword);

			float *dimWeight;
			cudaMemcpy(&dimWeight,&(devQueryInfo->depressed_dimWeight), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(dimWeight);

			float *upperBoundDist;
			cudaMemcpy(&upperBoundDist,&(devQueryInfo->upperBoundDist), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(upperBoundDist);

			float *lowerBoundDist;
			cudaMemcpy(&lowerBoundDist,&(devQueryInfo->lowerBoundDist), sizeof(float *), cudaMemcpyDeviceToHost);
			cudaFree(lowerBoundDist);


			cudaFree( devQueryInfo );
		}

}


template <class T>
void GPUManager::copyVectorFromHostToDevice(  host_vector<T>& hvec, device_vector<T> & _dvec )
{
	//_dvec = new device_vector<T>;
	//_dvec->reserve(hvec.size());

	//for ( int i = 0; i < hvec.size(); i++ )
	//{
	//	_dvec[i] = hvec[i];
		//_dvec->push_back(hvec[i]);
	//}

	_dvec = hvec;

}


void GPUManager::check_query_parameter(){

	if (!this->query_initialized)
	{
		cout << "query has not been initialized" << endl;
		exit(1);
	}

	if ( THREAD_PER_BLK < invert_list_spec_host.totalDimension )
	{
		cout << "thread per block must be larger than max dimension of x" << endl;
		exit(1);
	}
	if (this->invert_list_spec_host.numOfQuery > 65535)
	{
		cout << "number of query should not exceed 65535: need to modify the code" << endl;
		exit(1);
	}
	if (THREAD_PER_BLK > 1024)
	{
		cout << "THREAD_PER_BLK should not exceed 1024: need to modify the code" << endl;
		exit(1);
	}

}

void GPUManager::printPrunStatistics(){

	int sci = 0;
	for(int i=0;i<this->invert_list_spec_host.totalDimension;i++){
		int ci = thrust::count_if(d_query_feature.begin(),d_query_feature.end(),gCount(i));
		//cout<<"The item with count["<<i<<"] is:"<<ci<<endl;
		sci+=ci;
	}
	cout<<"the total item touched is:"<<sci<<endl;

	QueryFeatureEnt init;
	init.count=0;init.ACD =0.f;
	QueryFeatureEnt sum = thrust::reduce(d_query_feature.begin(),d_query_feature.end(),init,countFeature());
	cout<<"the average sum of count table (touched) per query is:"<< sum.count/d_query_info.size()<<endl;
	cout<<"the total number of items in the idnex is:"<<d_invert_list.size()<<endl;
	cout<<"prune rate (1-touched/total) ="<<1-((double)sum.count/d_query_info.size())/d_invert_list.size()<<endl;
}


void GPUManager::runTest() {
	int blk_num = 2;
	int topKValue = 7;

	host_vector<Result> h_result(THREAD_PER_BLK * blk_num);
	host_vector<Result> h_result_temp(THREAD_PER_BLK * blk_num);
	host_vector<int> h_query_result_count(blk_num);
	h_query_result_count[0] = THREAD_PER_BLK;
	h_query_result_count[1] = 2 * THREAD_PER_BLK;
	host_vector<int> h_valid_query;
	h_valid_query.resize(blk_num, 1);

	device_vector<Result> d_result(
			h_query_result_count[h_query_result_count.size() - 1]);
	device_vector<Result> d_result_temp(
			h_query_result_count[h_query_result_count.size() - 1]);
	device_vector<int> d_query_result_count(blk_num);
	device_vector<int> d_valid_query(blk_num);

	for (int i = 0; i < blk_num * THREAD_PER_BLK; i++) {

		h_result[i].count = i;	//rand()%(blk_num*THREAD_PER_BLK);
		h_result[i].lb = rand() % (blk_num * THREAD_PER_BLK);
		h_result[i].ub = rand() % (blk_num * THREAD_PER_BLK);

	}

	h_result[55].lb = -1;
	h_result[66].ub = -3;

	h_result[200].lb = -8;
	h_result[180].ub = -9;

	d_result = h_result;
	d_query_result_count = h_query_result_count;
	d_valid_query = h_valid_query;

	terminateCheck_kSelection_KernelPerQuery<<<blk_num, THREAD_PER_BLK / 5>>>(
			raw_pointer_cast(d_result.data()),
			raw_pointer_cast(d_result_temp.data()),
			raw_pointer_cast(d_query_result_count.data()),
			raw_pointer_cast(d_valid_query.data()), topKValue);

	h_result = d_result;

	for (int j = 0; j < blk_num; j++) {
		cout << "BLK =" << j << endl;
		for (int i = j * THREAD_PER_BLK; i < j * THREAD_PER_BLK + topKValue + 3;
				i++) {
			cout << " d_result_ub_sorted[" << h_result[i].count << "].ub ="
					<< h_result[i].ub;
		}
		cout << endl;

		for (int i = j * THREAD_PER_BLK; i < j * THREAD_PER_BLK + topKValue + 3;
				i++) {
			cout << " d_result_ub_sorted[" << h_result[i].count << "].lb ="
					<< h_result[i].lb;
		}
		cout << endl;

	}
}

//junk code

//==========================junk code  2014.06.10


/*

//================================== v1 of bi_direction_query_KernelPerQuery
*
 * each query take one block (kernel), therefore this function is query oriented parallel computing, this is an old version

bool GPUManager::bi_direction_query_KernelPerQuery ( int threshold, int topKValue, vector<Result>& _result )
{

	check_query_parameter();

	cudaProfilerStart();
	cudaEvent_t start, stop;
	cudaEvent_t gs,ge;
	float elapsedTime;
	float elapseTimeForCount;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&gs);
	cudaEventCreate(&ge);


	int terminate_sum;

	cudaEventRecord(gs, 0);
	cout << "GPUManager::bi_direction_query(): 0 -- compute_mapping_saving_pos_KernelPerQuery()" << endl;
	cudaEventRecord(start, 0);
	compute_mapping_saving_pos_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK, max_number_of_dimensions*sizeof(int2) >>> (
		raw_pointer_cast( this->d_query_info.data() ),
		raw_pointer_cast( this->d_invert_list.data() ),
		raw_pointer_cast( this->d_invert_list_idx.data() ),
		raw_pointer_cast( this->d_query_feature.data() ),//QueryFeatureEnt* query_feature, count&ACD table
		false,
		max_value_per_dimension,
		raw_pointer_cast( this->d_vec_indexDimensionEntry.data() )
		);
	 //cudaDeviceSynchronize();
	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&elapsedTime, start, stop);
	 exec_time[0] += elapsedTime;

	device_vector<int> threshold_count(invert_list_spec_host.numOfQuery * THREAD_PER_BLK, 0);	// prefix count for each thread
	device_vector<int> query_result_count(invert_list_spec_host.numOfQuery,0);				// prefix count for each block (i.e. each query)

	cout << "GPUManager::bi_direction_query(): 1 -- prefix_count_KernelPerQuery()" << endl;
	elapseTimeForCount = 0;
	cudaEventRecord(start, 0);
	prefix_count_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK>>>(
		raw_pointer_cast(this->d_query_feature.data()),
		raw_pointer_cast(threshold_count.data()),
		raw_pointer_cast(query_result_count.data()),
		threshold
		);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapseTimeForCount += elapsedTime;
	exec_time[7] += elapsedTime;

	cudaEventRecord(start, 0);
	thrust::inclusive_scan(threshold_count.begin(), threshold_count.end(), threshold_count.begin()); // per thread inclusive scan
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[8] += elapsedTime;
	elapseTimeForCount += elapsedTime;

	cudaEventRecord(start, 0);
	thrust::inclusive_scan(query_result_count.begin(), query_result_count.end(), query_result_count.begin()); // per block inclusive scan
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[9] += elapsedTime;
	elapseTimeForCount += elapsedTime;
	exec_time[1] += elapseTimeForCount;


	cout << "GPUManager::bi_direction_query(): 2 -- output_result_bidrection_search_KernelPerQuery()" << endl;
	device_vector<Result> d_result_ub_sorted(threshold_count[threshold_count.size()-1]);


	cudaEventRecord(start, 0);
	output_result_bidrection_search_KernelPerQuery<<< this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK, 3*max_number_of_dimensions*sizeof(float) >>>(
		raw_pointer_cast(this->d_query_feature.data()),
		raw_pointer_cast(this->d_query_info.data()),
		raw_pointer_cast(threshold_count.data()),
		raw_pointer_cast(d_result_ub_sorted.data()),//record the lower bound and upper bound
		threshold,
		raw_pointer_cast( this->d_vec_indexDimensionEntry.data() )
		);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[2] += elapsedTime;

	device_vector<Result> d_result_lb_sorted(d_result_ub_sorted.begin(),d_result_ub_sorted.end());


	cout << "GPUManager::bi_direction_query(): 3 -- sort d_result_lb & ub " << endl;
	cudaEventRecord(start, 0);
	thrust::sort( d_result_ub_sorted.begin(), d_result_ub_sorted.end(), Ubcomapre() );
	//Result_kSelectionSort()


	thrust::sort( d_result_lb_sorted.begin(), d_result_lb_sorted.end(), Lbcompare() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[3] += elapsedTime;


	cout << "GPUManager::bi_direction_query(): 4 -- terminate_check_KernelPerQuery()" << endl;
	cudaEventRecord(start, 0);
	terminateCheck_afterSort_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK>>>(
		raw_pointer_cast(d_result_ub_sorted.data()),
		raw_pointer_cast(d_result_lb_sorted.data()),
		raw_pointer_cast(query_result_count.data()),
		raw_pointer_cast(d_valid_query.data()),
		topKValue,
		this->num_of_rounds);
	terminate_sum = thrust::reduce(d_valid_query.begin(),d_valid_query.end());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[4] += elapsedTime;

	cout << "GPUManager::bi_direction_query(): 5 -- extract_topK_KernelPerQuery()" << endl;
	device_vector<Result> d_result( this->invert_list_spec_host.numOfQuery * topKValue );
	cudaEventRecord(start, 0);
	extract_topK_KernelPerQuery<<< this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK >>>(
		raw_pointer_cast(d_result_ub_sorted.data()),
		raw_pointer_cast(query_result_count.data()),
		raw_pointer_cast(d_result.data()),
		topKValue);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[5] += elapsedTime;


	cudaEventRecord(ge, 0);
	cudaEventSynchronize(ge);
	cudaEventElapsedTime(&elapsedTime, gs, ge);
	exec_time[6] += elapsedTime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(gs);
	cudaEventDestroy(ge);

	cudaProfilerStop();



//	for ( int index = 0 ; index < d_result_ub_sorted.size(); index ++ )
//	{
//		Result tempResult = d_result_ub_sorted[index];
//		tempResult.print_result_entry();
//		printf("\n");
//	}



	num_of_rounds++;

	_result.resize(d_result.size());
	HANDLE_ERROR( cudaMemcpy( &_result[0], raw_pointer_cast(d_result.data()), d_result.size()*sizeof(Result), cudaMemcpyDeviceToHost) );

	// no more query should be made to this batch
	if (terminate_sum == 0) {
		this->query_initialized = false;
	}


	return terminate_sum == 0;
}
*/









/**
 *
 *
bool GPUManager::bi_direction_query_KernelPerDim_V1 (int threshold, int topKValue, vector<Result>& _result )
{

	check_query_parameter();

	cudaProfilerStart();
	cudaEvent_t start, stop;
	cudaEvent_t gs, ge;
	float elapsedTime;
	float elapseTimeForCount;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&gs);
	cudaEventCreate(&ge);

	cout	<< "GPUManager::bi_direction_query(): 0 -- compute_mapping_saving_pos_KernelPerDim()"
			<< endl;
	cudaEventRecord(gs, 0);
	cudaEventRecord(start, 0);
	compute_mapping_saving_pos_KernelPerDim<<<sumQueryDims, THREAD_PER_BLK>>>(
			raw_pointer_cast(this->d_query_info.data()),
			raw_pointer_cast(this->d_invert_list.data()),
			raw_pointer_cast(this->d_invert_list_idx.data()),
			raw_pointer_cast(this->d_query_feature.data()),	//QueryFeatureEnt* query_feature, count&ACD table
			false, max_value_per_dimension,
			raw_pointer_cast(this->minDomainForAllDimension.data()),
			raw_pointer_cast(this->maxDomainForAllDimension.data()));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[0] += elapsedTime;

	cout	<< "GPUManager::bi_direction_query(): 1 -- prefix_count_KernelPerQuery()"
			<< endl;
	device_vector<int> threshold_count(	invert_list_spec_host.numOfQuery * THREAD_PER_BLK, 0);// prefix count for each thread
	device_vector<int> query_result_count(invert_list_spec_host.numOfQuery, 0);	// prefix count for each block (i.e. each query)
	elapseTimeForCount=0;

	cudaEventRecord(start, 0);
	depressed_prefix_count_KernelPerQuery_V1<<<this->invert_list_spec_host.numOfQuery,
			THREAD_PER_BLK>>>(raw_pointer_cast(this->d_query_feature.data()),
			raw_pointer_cast(threshold_count.data()),
			raw_pointer_cast(query_result_count.data()), threshold,
			raw_pointer_cast(this->minDomainForAllDimension.data()),
			raw_pointer_cast(this->maxDomainForAllDimension.data()));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapseTimeForCount+=elapsedTime;
	exec_time[7] += elapsedTime;

	cudaEventRecord(start, 0);
	thrust::inclusive_scan(threshold_count.begin(), threshold_count.end(),
			threshold_count.begin()); // per thread inclusive scan
	cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		elapseTimeForCount+=elapsedTime;
		exec_time[8] += elapsedTime;
	cudaEventRecord(start, 0);
	thrust::inclusive_scan(query_result_count.begin(), query_result_count.end(),
			query_result_count.begin()); // per block inclusive scan
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapseTimeForCount+=elapsedTime;
	exec_time[9] += elapsedTime;
	exec_time[1] += elapseTimeForCount;

	cout	<< "GPUManager::bi_direction_query(): 2 -- output_result_bidrection_search_KernelPerQuery()"
			<< endl;
	device_vector<Result> d_result_ub_sorted(
			threshold_count[threshold_count.size() - 1]);

	cudaEventRecord(start, 0);
	output_result_bidrection_search_KernelPerQuery_V1<<<
			this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK,
			3 * max_number_of_dimensions*sizeof(float)>>>(
			raw_pointer_cast(this->d_query_feature.data()),
			raw_pointer_cast(this->d_query_info.data()),
			raw_pointer_cast(threshold_count.data()),
			raw_pointer_cast(d_result_ub_sorted.data()), //record the lower bound and upper bound
			threshold, raw_pointer_cast(this->minDomainForAllDimension.data()),
			raw_pointer_cast(this->maxDomainForAllDimension.data()));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[2] += elapsedTime;

	cout << "GPUManager::bi_direction_query(): 3 -- sort d_result_lb & ub"
			<< endl;
	device_vector<Result> d_result_lb_sorted(d_result_ub_sorted.begin(),
			d_result_ub_sorted.end());
	cudaEventRecord(start, 0);
	thrust::sort(d_result_ub_sorted.begin(), d_result_ub_sorted.end(),
			Ubcomapre());
	thrust::sort(d_result_lb_sorted.begin(), d_result_lb_sorted.end(),
			Lbcompare());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[3] += elapsedTime;

	cout
			<< "GPUManager::bi_direction_query(): 4 -- terminate_check_KernelPerQuery()"
			<< endl;
	cudaEventRecord(start, 0);
	terminate_check_KernelPerQuery_V1<<<this->invert_list_spec_host.numOfQuery,
			THREAD_PER_BLK>>>(raw_pointer_cast(d_result_ub_sorted.data()),
			raw_pointer_cast(d_result_lb_sorted.data()),
			raw_pointer_cast(query_result_count.data()),
			raw_pointer_cast(d_valid_query.data()), topKValue,
			this->num_of_rounds);
	int terminate_sum = thrust::reduce(d_valid_query.begin(),
			d_valid_query.end());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[4] += elapsedTime;

	cout
			<< "GPUManager::bi_direction_query(): 4 -- extract_topK_KernelPerQuery()"
			<< endl;
	device_vector<Result> d_result(
			this->invert_list_spec_host.numOfQuery * topKValue);
	cudaEventRecord(start, 0);
	extract_topK_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,
			THREAD_PER_BLK>>>(raw_pointer_cast(d_result_ub_sorted.data()),
			raw_pointer_cast(query_result_count.data()),
			raw_pointer_cast(d_result.data()), topKValue);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[5] += elapsedTime;

	cudaEventRecord(ge, 0);
	cudaEventSynchronize(ge);
	cudaEventElapsedTime(&elapsedTime, gs, ge);
	exec_time[6] += elapsedTime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaProfilerStop();

	num_of_rounds++;

	_result.resize(d_result.size());
	HANDLE_ERROR(
			cudaMemcpy(&_result[0], raw_pointer_cast(d_result.data()),
					d_result.size() * sizeof(Result), cudaMemcpyDeviceToHost));

	// no more query should be made to this batch
	if (terminate_sum == 0) {
		this->query_initialized = false;
	}

	return terminate_sum == 0;
}
 *
 *
 *
 */


/**
 *
 *

//================================================= V2 of bi_direction_query_KernelPerQuery
**
 * each query take one block (kernel), therefore this function is query oriented parallel computing, this is an old version
 *
bool GPUManager::bi_direction_query_KernelPerQuery_V2 ( int threshold, int topKValue, vector<Result>& _result )
{

	check_query_parameter();

	cudaProfilerStart();
	cudaEvent_t start, stop;
	cudaEvent_t gs,ge;
	float elapsedTime;
	float elapseTimeForCount;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&gs);
	cudaEventCreate(&ge);


	int terminate_sum;

	cudaEventRecord(gs, 0);
	cout << "GPUManager::bi_direction_query(): 0 -- compute_mapping_saving_pos_KernelPerQuery()" << endl;
	cudaEventRecord(start, 0);
	compute_mapping_saving_pos_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK, max_number_of_dimensions*sizeof(int2) >>> (
		raw_pointer_cast( this->d_query_info.data() ),
		raw_pointer_cast( this->d_invert_list.data() ),
		raw_pointer_cast( this->d_invert_list_idx.data() ),
		raw_pointer_cast( this->d_query_feature.data() ),//QueryFeatureEnt* query_feature, count&ACD table
		false,
		max_value_per_dimension,
		raw_pointer_cast( this->minDomainForAllDimension.data() ),
		raw_pointer_cast( this->maxDomainForAllDimension.data() ) );
	 //cudaDeviceSynchronize();
	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&elapsedTime, start, stop);
	 exec_time[0] += elapsedTime;



	device_vector<int> threshold_count(invert_list_spec_host.numOfQuery * THREAD_PER_BLK, 0);	// prefix count for each thread
	device_vector<int> query_result_count(invert_list_spec_host.numOfQuery,0);				// prefix count for each block (i.e. each query)

	cout << "GPUManager::bi_direction_query(): 1 -- prefix_count_KernelPerQuery()" << endl;
	elapseTimeForCount = 0;
	cudaEventRecord(start, 0);
	prefix_count_KernelPerQuery<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK,THREAD_PER_BLK*sizeof(int)>>>(
		raw_pointer_cast(this->d_query_feature.data()),
		raw_pointer_cast(threshold_count.data()),
		raw_pointer_cast(query_result_count.data()),
		threshold,
		raw_pointer_cast( this->minDomainForAllDimension.data() ),
		raw_pointer_cast( this->maxDomainForAllDimension.data() ) );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	elapseTimeForCount += elapsedTime;
	exec_time[7] += elapsedTime;

	cudaEventRecord(start, 0);
	thrust::inclusive_scan(threshold_count.begin(), threshold_count.end(), threshold_count.begin()); // per thread inclusive scan
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[8] += elapsedTime;
	elapseTimeForCount += elapsedTime;

	cudaEventRecord(start, 0);
	thrust::inclusive_scan(query_result_count.begin(), query_result_count.end(), query_result_count.begin()); // per block inclusive scan
	//int query_result_count_idx = THREAD_PER_BLK -1;
	//for(int i=0;i<query_result_count.size();i++){
		//cout<<"previous query_result_count "<<i<<" "<<query_result_count[i]<<endl;
	//	query_result_count[i] = threshold_count[query_result_count_idx];
	//	query_result_count_idx+=THREAD_PER_BLK;
		//cout<<"after query_result_count "<<i<<" "<<query_result_count[i]<<endl;
	//}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[9] += elapsedTime;
	elapseTimeForCount += elapsedTime;
	exec_time[1] += elapseTimeForCount;


	cout << "GPUManager::bi_direction_query(): 2 -- output_result_bidrection_search_KernelPerQuery()" << endl;
	device_vector<Result> d_result_ub_sorted(threshold_count[threshold_count.size()-1]);
	cudaEventRecord(start, 0);
	//note: v1 version is just OK
	output_result_bidrection_search_KernelPerQuery_V1<<< this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK, 3*max_number_of_dimensions*sizeof(float) >>>(
		raw_pointer_cast(this->d_query_feature.data()),
		raw_pointer_cast(this->d_query_info.data()),
		raw_pointer_cast(threshold_count.data()),
		//raw_pointer_cast(query_result_count.data()),
		raw_pointer_cast(d_result_ub_sorted.data()),//record the lower bound and upper bound
		threshold,
		raw_pointer_cast( this->minDomainForAllDimension.data() ),
		raw_pointer_cast( this->maxDomainForAllDimension.data() ) );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[2] += elapsedTime;

	device_vector<Result> d_result_lb_sorted(d_result_ub_sorted.begin(),d_result_ub_sorted.end());


	cout << "GPUManager::bi_direction_query(): 3 -- sort d_result_lb & ub " << endl;
	cudaEventRecord(start, 0);
	//for(int i=0;i<this->invert_list_spec_host.numOfQuery;i++){
	///int block_start =(i == 0 ? 0 : query_result_count[i - 1]);
	//int block_end = query_result_count[i];
	//thrust::sort( d_result_ub_sorted.begin()+ block_start, d_result_ub_sorted.begin() +block_end, Ubcomapre() );
	//thrust::sort( d_result_lb_sorted.begin()+block_start, d_result_lb_sorted.begin() +block_end, Lbcompare() );
	//}

	thrust::sort( d_result_ub_sorted.begin(), d_result_ub_sorted.end() , Ubcomapre() );
	thrust::sort( d_result_lb_sorted.begin(), d_result_lb_sorted.end() , Lbcompare() );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[3] += elapsedTime;


	cout << "GPUManager::bi_direction_query(): 4 -- terminate_check_KernelPerQuery()" << endl;
	cudaEventRecord(start, 0);
	terminate_check_KernelPerQuery_V1<<<this->invert_list_spec_host.numOfQuery,THREAD_PER_BLK>>>(
		raw_pointer_cast(d_result_ub_sorted.data()),
		raw_pointer_cast(d_result_lb_sorted.data()),
		raw_pointer_cast(query_result_count.data()),
		raw_pointer_cast(d_valid_query.data()),
		topKValue,
		this->num_of_rounds);
	terminate_sum = thrust::reduce(d_valid_query.begin(),d_valid_query.end());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[4] += elapsedTime;

	cout << "GPUManager::bi_direction_query(): 5 -- extract_topK_KernelPerQuery()" << endl;
	device_vector<Result> d_result( this->invert_list_spec_host.numOfQuery * topKValue );
	cudaEventRecord(start, 0);
	extract_topK_KernelPerQuery<<< this->invert_list_spec_host.numOfQuery, THREAD_PER_BLK >>>(
		raw_pointer_cast(d_result_ub_sorted.data()),
		raw_pointer_cast(query_result_count.data()),
		raw_pointer_cast(d_result.data()),
		topKValue);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	exec_time[5] += elapsedTime;


	cudaEventRecord(ge, 0);
	cudaEventSynchronize(ge);
	cudaEventElapsedTime(&elapsedTime, gs, ge);
	exec_time[6] += elapsedTime;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaEventDestroy(gs);
	cudaEventDestroy(ge);

	cudaProfilerStop();

	num_of_rounds++;

	_result.resize(d_result.size());
	HANDLE_ERROR( cudaMemcpy( &_result[0], raw_pointer_cast(d_result.data()), d_result.size()*sizeof(Result), cudaMemcpyDeviceToHost) );

	// no more query should be made to this batch
	if (terminate_sum == 0) {
		this->query_initialized = false;
	}


	return terminate_sum == 0;
}

 */
