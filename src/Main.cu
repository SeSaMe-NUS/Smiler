
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


#include "tools/WrapperIndexBuilder.h"
#include "tools/BladeLoader.h"
#include "tools/DataOperator/DataProcess.h"
#include "tools/WrapperDataProcess.h"
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
