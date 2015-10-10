/*
 * SmilerDemo.cpp
 *
 *  Created on: Oct 8, 2014
 *      Author: zhoujingbo
 */

#include "SmilerDemo.h"

#include "../searcher/WrapperTSProcess.h"

#include "../searcher/WrapperScan.h"

#include "../smilerManager/WrapperTSManagerOnGpuIdx.h"



SmilerDemo::SmilerDemo() {
	// TODO Auto-generated constructor stub

}

SmilerDemo::~SmilerDemo() {
	// TODO Auto-generated destructor stub
}


void SmilerDemo::runExp_TSLOOCVContPred_errorAndTime(string fileHolder, int fcol_start, int fcol_end, int queryNumPerBlade){

	WrapperTSManagerOnGpuIdx wtsm;//


	//start parameter setting
	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	int contPrdStep = 20;//continuous prediction step
	int queryLenghLoaded = 1000; // queryLenghLoaded+contPrdStep is equal to the length of the cut-out time series segment
	int ELV[]={32,64,96};//Ensemble Length Vector
	int EKV[] ={8,16,32};//Ensemble kNN Vector
	int y_step_ahead_array[]={20};//y: step ahead prediction, It is an array for multiple steps ahead prediction
	int sc_band = 8;//for LB_keogh lower bound parameter
	int windowDim = 16;//window size
	double range = 5, sill = 1, nugget = 1;//seed parameters for Gaussian Processes
	bool selfCorr = true;//
	//end parameter setting

	vector<int> Lvec(ELV,ELV+sizeof(ELV)/sizeof(int));
	vector<int> Kvec(EKV,EKV+sizeof(EKV)/sizeof(int));


	cout<<"//++++++++++++++++++++++++++++++++++++++++++++++++++"<<endl;
	cout<<"//++++++++++++++++++++++++runExp_TSLOOCVContPred_errorAndTime()"<<endl;
	cout<<"//++++++++++++++++++++++++ test disable selfCorr"<<endl;
	cout<<" selfCorr="<<selfCorr<<endl;
	cout<<"//++++++++++++++++++varying y_step_ahead_array={";
	for(int i=0;i<sizeof(y_step_ahead_array)/sizeof(int);i++){
			cout<<" "<<y_step_ahead_array[i]<<",";
	}
	cout<<"}"<<endl;

	cout<<"file:"<<fileHolder<<endl;
	cout<<" fcol_start:"<<fcol_start<<" fcol_end:"<<fcol_end<<" queryNumPerBlade:"<<queryNumPerBlade<<endl;


	cout<<" sc_band:"<<sc_band<<" contPrdStep:"<<contPrdStep<< " queryLenghLoaded:"<<queryLenghLoaded<<" window dim:"<<windowDim<<endl;
	cout<<" seed parameter: range:"<<range<<" sill"<<sill<<" nugget:"<<nugget<<endl;
	cout<<" query_item_vec vec:{";
	for(int i=0;i<sizeof(ELV)/sizeof(int);i++){
					cout<<" "<<ELV[i]<<",";
		}
	cout<<" }"<<endl;

	cout<<" topk vec:{";
	for(int i=0;i<sizeof(EKV)/sizeof(int);i++){
		cout<<" "<<EKV[i]<<" ,";
	}
	cout<<" }"<<endl;



	for (int i = 0; i < sizeof(y_step_ahead_array) / sizeof(int); i++) {
		cout<<"//=================================================="<<endl;
		cout<<" compute topk with y_step_ahead ["<<i<<"]:"<<y_step_ahead_array[i]<<endl;
		cout<<"//==============start runrunMain_TSLOOCVContPred_mulSensors()"<<endl;
		wtsm.runMain_TSLOOCVContPred_mulSensors(fileHolder, fcol_start,
				fcol_end, queryNumPerBlade,
				contPrdStep, queryLenghLoaded,
				Lvec, Kvec, range,
				sill, nugget, y_step_ahead_array[i], sc_band, selfCorr);// main function, this function loads data, cuts queries from the data,
																		//and then run the smiler predictor(with kNN search and GP)
	}
}

void runExp_TSLOOCVContPred_errorAndTime_isp(){
	string fileHolder = "data/isp/isp_normal";

	int fcol_start = 1;
	int fcol_end = 1;
	int queryNumPerBlade = 100;//can be set as 1024 or larger, this is to simulate multiple sensor prediction. If there have already been multiple sensors in the fileHoder, just set this queryNumPerBlade = 1

	SmilerDemo tse;
	tse.runExp_TSLOOCVContPred_errorAndTime(fileHolder,  fcol_start,  fcol_end,  queryNumPerBlade);
}


void SmilerDemo::runDemo(){

	runExp_TSLOOCVContPred_errorAndTime_isp();

}
