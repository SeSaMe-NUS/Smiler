/*
 * WrapperDataProcess.cpp
 *
 *  Created on: Aug 22, 2014
 *      Author: zhoujingbo
 */

#include "WrapperDataProcess.h"

WrapperDataProcess::WrapperDataProcess() {
	// TODO Auto-generated constructor stub

}

WrapperDataProcess::~WrapperDataProcess() {
	// TODO Auto-generated destructor stub
}


void WrapperDataProcess::run_ZNormalize(){
	DataProcess dp;
	//dp.z_normalizeData("data/calit2/CalIt2_7.csv", 1,"data/calit2/CalIt2_7_normal.csv",3);
	//dp.z_normalizeData("data/Dodgers/dodgers_clean.csv", 1,"data/Dodgers/dodgers_clean_normal.csv",1);
	//dp.z_normalizeData("data/isp/isp.csv", 1,"data/isp/isp_normal.csv");
	//dp.z_normalizeData("data/water/jetta.csv", 1,"data/water/jetta_salinity_normal.csv",1);
	//dp.z_normalizeData("data/water/jetta_salinity_normal.csv", 2,"data/water/jetta_ec_normal.csv",1);
	//dp.z_normalizeData("data/water/jetta_ec_normal.csv", 3,"data/water/jetta_normal.csv",1);

	//dp.z_normalizeData("data/lta/lta_carParksLots.csv", 10,"data/lta/lta_carParksLots_normal_10.csv");
	dp.z_normalizeData("data/lta/lta_carParksLots.csv", 1, 26,"data/lta/lta_carParksLots_normal.csv");


}

void WrapperDataProcess::run_RemoveMissing(){
	DataProcess dp;
	dp. remove_missData("data/Dodgers/Dodgers.csv",  1, "data/Dodgers/dodgers_clean.csv");
}
