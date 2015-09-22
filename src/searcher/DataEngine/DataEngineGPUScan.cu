/*
 * DataEngineGPUScan.cpp
 *
 *  Created on: Jun 25, 2014
 *      Author: zhoujingbo
 */

#include "DataEngineGPUScan.h"

//DataEngine_GPUScan::DataEngine_GPUScan() {
//	// TODO Auto-generated constructor stub
//
//}
//
//DataEngine_GPUScan::~DataEngine_GPUScan() {
//	// TODO Auto-generated destructor stub
//}
//


void DataEngine_GPUScan_helper::inclusiveScan(device_vector<int>& d_vec_size, device_vector<int>& d_vec_inclusive){

	thrust::inclusive_scan(d_vec_size.begin(),d_vec_size.end(),d_vec_inclusive.begin());


}
