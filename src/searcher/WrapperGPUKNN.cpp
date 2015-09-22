/*
 * WrapperGPUKNN.cpp
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#include <sstream>
using namespace std;


#include "WrapperGPUKNN.h"
#include "GPUKNN/UtlGPU.h"

#include "GPUKNN/generalization.h"
#include "GPUKNN/GPUManager.h"


WrapperGPUKNN::WrapperGPUKNN()
{
	// TODO Auto-generated constructor stub

}

WrapperGPUKNN::~WrapperGPUKNN()
{
	// TODO Auto-generated destructor stub
}


 int WrapperGPUKNN::runOnSimpleShift()
 {


	GPUSpecification test_spec;
	test_spec.numOfDocToExpand = 1000;
	test_spec.totalDimension = 128;

	//test_spec.invertedListPath = "../data/sift_exp_data/sift_small";
	//test_spec.invertedIndexLengthFile = "../data/sift_exp_data/sorted.txt";

	test_spec.invertedListPath = "../data/sift_4m/inverted_index.bin";
	test_spec.indexDimensionEntry.resize(test_spec.totalDimension);
	for(int i = 0; i < test_spec.totalDimension; i++){
		test_spec.indexDimensionEntry[i].minDomain = 0;            // dangerous:!!
		test_spec.indexDimensionEntry[i].maxDomain = 255;            // dangerous:!!
		test_spec.indexDimensionEntry[i].bucketWidth = 1;            // dangerous:!!

	}

	int topK = 5;

	clock_t start = clock();

	vector<GpuWindowQuery> test_query;

	GPUManager test_manager(test_spec,0);
	test_manager.init_GPU_query(test_query);

	//test_manager.print_invert_list();
	//test_manager.print_query();



	double total_time = 0;

	vector<Result> temp_result;
	bool finished = false;
	while(!finished)
	{
		finished = test_manager.bi_direction_query_KernelPerDim( 0, topK, temp_result );

		double time = (double)(clock() - start) / CLOCKS_PER_SEC;
		total_time += time;
		cout << "iteration takes time: " << time << endl;

		test_manager.print_result_vec(temp_result);

	}

	cout << "finished with total time : " << total_time << endl;


	//part 0: expansion, update count and ACD; part 1: compute lb and ub for features whose count > 0; part 2: k candidates selection (by sorting in current implementation); part 3: terminate condition check
	cout << "Profiling the time : " << endl;
		for(int i = 0; i < test_manager.exec_time.size(); i++){
			cout << "part " << i << " takes " << test_manager.exec_time[i] << endl;
		}

	//vector<Result> temp_result;
	//test_manager.point_query(temp_result,5);
	//print_result_vec(temp_result);

	//test_manager.print_query_doucment_mapping();
	return 0;
}



 int WrapperGPUKNN::runOnIntegerDataFile(){

	GPUSpecification test_spec;
	test_spec.numOfDocToExpand = 100;
	test_spec.totalDimension = 32;
	int queryNum = 16;
	float lp = 2;
	float upwardDistBound = 0;
	float downwardDistbound = 0;
	int topK = 5;
	//string dataHolder = "data/calit2/CalIt2_7";
	string dataHolder = "data/Dodgers/Dodgers";


	cout << "WrapperGPUKNN::runOnCalit2()"<< endl;

	std::stringstream idxPath;

	//idxPath<<"data/calit2/CalIt2_7_d"<<test_spec.totalDimension<<"_sw.idx";
	idxPath<<dataHolder<<"_d"<<test_spec.totalDimension<<"_sw.idx";

	test_spec.invertedListPath = idxPath.str().c_str();

	test_spec.indexDimensionEntry.resize(test_spec.totalDimension);

	std::stringstream queryFileStream;
	//queryFileStream<<"data/calit2/CalIt2_7_d"<<test_spec.totalDimension<<"_q16_dir.query";
	queryFileStream<<dataHolder<<"_d"<<test_spec.totalDimension<<"_q"<<queryNum<<"_dir.query";
	string queryFile= queryFileStream.str().c_str();

	for(int i = 0; i < test_spec.totalDimension; i++){

		test_spec.indexDimensionEntry[i].minDomain = 0;            // dangerous:!!
		test_spec.indexDimensionEntry[i].maxDomain = 127;            // dangerous:!!
		test_spec.indexDimensionEntry[i].bucketWidth = 1;            // dangerous:!!


	}


	GPUManager test_manager(test_spec, 0);
	test_manager.set_query_DefaultDisFuncType(lp);
	test_manager.set_query_DefaultDistBound(upwardDistBound,downwardDistbound);
	vector<GpuWindowQuery> querySet;

	test_manager.readQueryFromFile(queryFile, querySet);
	test_manager.init_GPU_query(querySet);

	double total_time = 0;

	vector<Result> temp_result;
	bool finished = false;
	int iteration=0;

	int count =0;
	while (!finished )
	{

		clock_t start = clock();

		finished = test_manager.bi_direction_query_KernelPerDim( 0, topK, temp_result);


		clock_t end = clock();
		double time = (double)(end - start) / CLOCKS_PER_SEC;
		total_time += time;
		iteration++;
		cout << "iteration "<< iteration <<" takes time: " << time << endl;

		test_manager.print_result_vec(temp_result);
	}

	test_manager.printPrunStatistics();

	string label;

		label = "bi_direction_query_KernelPerDim()";

	cout << label<<" with dimension = "<<test_spec.totalDimension<< " query number = "<<querySet.size()<<endl;
	cout<<" finished with total time : " << total_time <<" with "<<iteration<<" iterations"<< endl;


	//part 0: expansion, update count and ACD; part 1: compute lb and ub for features whose count > 0; part 2: k candidates selection (by sorting in current implementation); part 3: terminate condition check
	cout << "Profiling the time : " << endl;
	for(int i = 0; i < test_manager.exec_time.size(); i++){
		cout << "part " << i << " takes " << test_manager.exec_time[i] << endl;
	}

	//vector<Result> temp_result;
	//test_manager.point_query(temp_result,5);
	//print_result_vec(temp_result);

	//test_manager.print_query_doucment_mapping();


	return 0;

}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int WrapperGPUKNN::runGPUKNN(void) {

	runOnIntegerDataFile();
	return 0;

}


