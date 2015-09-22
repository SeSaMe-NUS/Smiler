/*
 * WrapperDataProcess.h
 *
 *  Created on: Aug 22, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERDATAPROCESS_H_
#define WRAPPERDATAPROCESS_H_

#include "DataOperator/DataProcess.h"
#include <string>
using namespace std;

class WrapperDataProcess {
public:
	WrapperDataProcess();
	virtual ~WrapperDataProcess();

	void run_ZNormalize();

	void run_RemoveMissing();
};

#endif /* WRAPPERDATAPROCESS_H_ */
