/*
 * ParabalaOpt.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: zhoujingbo
 */

#include "ParabalaOpt.h"

#include "../gp/gaussianProcesses/ParabalaFunc.h"
#include "../gp/optimisation/SCGModelTrainer.h"

#include <iostream>
using namespace std;

ParabalaOpt::ParabalaOpt() {
	// TODO Auto-generated constructor stub

}

ParabalaOpt::~ParabalaOpt() {
	// TODO Auto-generated destructor stub
}

void ParabalaOpt::testParalaba(){


	ParabalaFunc pf;


	SCGModelTrainer scggp(pf);
	scggp.setDisplay(true);
	scggp.Train(10);//UtlPredictor_namespace::trainStep);//for exp

   cout<<" para:"<<pf.getParametersVector()(0)<<endl;

}
