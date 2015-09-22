/*
 * SmilerDemo.h
 *
 *  Created on: Oct 8, 2014
 *      Author: zhoujingbo
 */

#ifndef SMILERDEMO_H_
#define SMILERDEMO_H_


#include <string>

class SmilerDemo {
public:
	SmilerDemo();
	virtual ~SmilerDemo();

	void runDemo();

	void runExp_TSLOOCVContPred_errorAndTime(std::string fileHolder, int fcol_start, int fcol_end, int queryNumPerBlade);
};

#endif /* TSEXPERIMENT2015_H_ */
