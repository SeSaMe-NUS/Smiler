/*
 * UtlDataEngine.h
 *
 *  Created on: Jun 25, 2014
 *      Author: zhoujingbo
 */

#ifndef UTLDATAENGINE_H_
#define UTLDATAENGINE_H_

#include <vector>
#include <iostream>
#include <string>
using namespace std;

//class UtlDataEngine {
//public:
//	UtlDataEngine();
//	virtual ~UtlDataEngine();
//
//};


namespace UtlDataEngine{
//static int igonre_step = 0;//if ignore step == 1, the first element of the kNN search is ignored since it is the query element themselves if not leaving out
}

template <class T>
struct XYtrnPerGroup{

	vector<vector<vector<T> > > Xtrn;//xtrn for one group query with different dimensions
	vector<vector<T> > Ytrn;//Ytrn for one group query with different dimensions
	vector<vector<float> > dist;//distance from training data point to query data point

	void resize(int size){
		Xtrn.resize(size);
		Ytrn.resize(size);
		dist.resize(size);
	}

	void print(){
		cout<<"number of train set:"<<Xtrn.size()<<endl;
		for(int i=0;i<Xtrn.size();i++){
			cout<<"train set:"<<i<<endl;
			for(int j=0;j<Xtrn[i].size();j++){

				cout<<"train item["<<i<<"]["<<j<<"]"<<endl;
				for(int k=0;k<Xtrn[i][j].size();k++){
					cout<<" "<<Xtrn[i][j][k];
				}
				cout<<"; observation: "<<Ytrn[i][j]<<"; dist:"<<dist[i][j]<<endl;
			}
		}
	}

};

struct PredictionResultSet{

	vector<vector<double> > meanMat;
	vector<vector<double> > varMat;
	vector<vector<double> >  weightMat;
	double tst;

};
#endif /* UTLDATAENGINE_H_ */
