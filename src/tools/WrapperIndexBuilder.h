/*
 * WrapperIndexBuilder.h
 *
 *  Created on: Jun 13, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERINDEXBUILDER_H_
#define WRAPPERINDEXBUILDER_H_
#include <string>

using namespace std;
class WrapperIndexBuilder {
public:
	WrapperIndexBuilder();
	virtual ~WrapperIndexBuilder();

	//entrance function
	int runBuilderIndex();


	int runSingleIndexBuilder(void);
	int  runSingleIndexBuilder_int(string dataHolder,int fcol,int totalDimension, int queryNum, string winType,int bits_for_value);


	int runGroupIndexBuilder_IntegerData(void);
	//void runIndexBuilder();//entrance function

};

#endif /* WRAPPERINDEXBUILDER_H_ */
