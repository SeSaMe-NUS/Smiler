/*
 * WrapperIndexBuilder.cpp
 *
 *  Created on: Jun 13, 2014
 *      Author: zhoujingbo
 */

#include "WrapperIndexBuilder.h"
#include "DataOperator/invListBuilder.h"
#include "DataOperator/DataProcess.h"
#include "BladeLoader.h"



#include <sstream>
using namespace std;


WrapperIndexBuilder::WrapperIndexBuilder() {
	// TODO Auto-generated constructor stub

}

WrapperIndexBuilder::~WrapperIndexBuilder() {
	// TODO Auto-generated destructor stub
}


int WrapperIndexBuilder:: runSingleIndexBuilder_int(string dataHolder,int fcol,int dimensionNumber, int queryNum, string winType,int bits_for_value){


	std::stringstream queryFileStream;
	queryFileStream << dataHolder<<"_d" << dimensionNumber
			<< "_q"<< queryNum<<"_dir.query";

	//queryFileStream<<"data/Dodgers/Dodgers_d"<<totalDimension<<"_q"<<queryNum<<"_dir.query";
	string queryFile = queryFileStream.str().c_str();

	std::stringstream idxPath;
	idxPath << dataHolder<<"_d" << dimensionNumber << "_" << winType
			<<"w.idx";
	//idxPath << "data/Dodgers/Dodgers_d" << totalDimension<< "_sw.idx";
	string invertedListPath = idxPath.str().c_str();
	string dataFile = dataHolder+".csv";

	invListBuilder ilB;


	ilB.runBuild_IdxAndQuery(dataFile,fcol, "i",invertedListPath,queryFile,queryNum,dimensionNumber,bits_for_value, winType);
	return 0;
}

int WrapperIndexBuilder::runSingleIndexBuilder(void) {

	//============this is for seting the index and query
	string dataHolder = "data/Dodgers/Dodgers";
	int fcol = 1;

	//string dataHolder = "data/calit2/CalIt2_7";
	//int fcol = 3;

	int dimensionNumber = 128;
	int queryNum = 8192;


	string winType = "s";
	int bits_for_value = 7;
	//===================== end for seting the index and query

	runSingleIndexBuilder_int(dataHolder, fcol, dimensionNumber,  queryNum,  winType, bits_for_value);


	return 0;
}

//
//int WrapperIndexBuilder::runGroupIndexBuilder_IntegerData(void){
//
//	//string dataHolder = "data/test/sequenceTest";
//	//string dataHolder = "data/test/seqTest_8k";
//	string dataHolder = DATA_HOLDER;// "data/Dodgers/Dodgers";
//	int fcol = DATA_HOLDER_COL;//1;
//
//	invListBuilder ilB;
//
//	//int totalDimension = 64;
//	int winDim = WINDOW_DIMENSION;//4;
//	int groupQueryNum = GROUPQUERY_NUM;//2;
//	int bladeNum = groupQueryNum;
//	//int totalDimension = winDim*groupQueryNum;
//	int queryLen = QUERY_MAX_LEN;//16;
//	string winType = "d";
//	int bits_for_value = 7;
//
//	std::stringstream queryFileStream;
//	queryFileStream <<dataHolder <<"_ql" << queryLen << "_gqn"
//				<< groupQueryNum << "_group.query";
//	string queryFile = queryFileStream.str();
//
//	std::stringstream idxPath;
//	idxPath << dataHolder<<"_wd" << winDim << "_bld" << bladeNum
//				<< "_bv" << bits_for_value << "_" << winType << "w_group.idx";
//	string invertedListPath = idxPath.str();
//
//	//get query from data files
//	string bladeDataFile = dataHolder+".csv";
//	ilB.runBuildPesudoGroupIdx(bladeDataFile.c_str(), fcol, "i",
//				invertedListPath, queryFile, bladeNum, groupQueryNum, queryLen,
//				winDim, bits_for_value, winType,false);
//
//	return 0;
//}
//

int WrapperIndexBuilder::runBuilderIndex(){
	runSingleIndexBuilder();
}

