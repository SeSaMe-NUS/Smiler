/*
 * WrapperGPUKNN.h
 *
 *  Created on: Apr 1, 2014
 *      Author: zhoujingbo
 */

#ifndef WRAPPERGPUKNN_H_
#define WRAPPERGPUKNN_H_

class WrapperGPUKNN {
public:
	WrapperGPUKNN();
	virtual ~WrapperGPUKNN();

public:
	 int runOnIntegerDataFile();
	 int runOnSimpleShift();
	 int runGPUKNN(void);
};

#endif /* WRAPPERGPUKNN_H_ */
