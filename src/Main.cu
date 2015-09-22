/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include "tools/WrapperIndexBuilder.h"
#include "tools/BladeLoader.h"
#include "tools/DataOperator/DataProcess.h"
#include "tools/WrapperDataProcess.h"
#include "tools/deviceDetector/deviceDetector.h"
#include "searcher/WrapperGPUKNN.h"
#include "searcher/WrapperScan.h"
#include "searcher/GPUKNN/GPUManager.h"
#include "searcher/TSProcess/TSProcessor.h"
#include "searcher/WrapperTSProcess.h"
#include "predictor/GP/WrapperPredictor.h"


#include "demo/SmilerDemo.h"



int main(void) {

	SmilerDemo sd;
	sd.runDemo();

	return 0;
}
