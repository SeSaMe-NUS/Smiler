/*
 * UtlPredictor.cpp
 *
 *  Created on: Apr 7, 2014
 *      Author: zhoujingbo
 */

#include "UtlPredictor.h"


namespace UtlPredictor_namespace{
void printMatrix(mat data) {
	//comment by jingbo
	//for (int i = 0; i < data.rows(); i++) {
	//	for (int j = 0; j < data.cols(); j++)
	for (uint i = 0; i < data.n_rows; i++) {
		for (uint j = 0; j < data.n_cols; j++)
			cerr << " " << data(i, j);
		cerr << endl;
	}
}

void printVector(vec data){
	for(uint i=0;i<data.n_rows;i++){
		cerr<<" "<<data(i);
	}
	cerr<<endl;
}

}

