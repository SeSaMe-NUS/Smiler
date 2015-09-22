/*
 * Util.cpp
 *
 *  Created on: Feb 22, 2014
 *      Author: zhoujingbo
 */

#include "UtlScan.h"



template<class T>
void printVector(vector<T>& data){
	cout<<"print vetor. len:"<<data.size()<<endl;
	for(int i=0;i<data.size();i++){
		cout<<"id: "<<i<<" value:"<<data[i]<<endl;
	}
	cout<<"element of vec printed:"<<data.size()<<endl;
	cout<<endl;
}



template <class T>
UtlDTW<T>::UtlDTW()
{
	// TODO Auto-generated constructor stub

}
template <class T>
UtlDTW<T>::~UtlDTW()
{
	// TODO Auto-generated destructor stub
}





template <class T>
int* UtlDTW<T>::selectMinK(int k, T* data, int s, int e){

	int* index = new int[k];
	T* dist = new T[k];

	for(int i=s;i<e;i++){

		T d = data[i];

		T maxd = -1; int idx = -1;
		for(int r=0;i<k;r++){
			if(dist[r]>=maxd){
				maxd = dist[r];
				idx = r;
			}
		}

		if(d<=maxd){
			dist[idx]=d;
			index[idx]=i;
		}

	}
	delete[] dist;
	return index;
}

template <class T>
 void UtlDTW<T>::printIntArray(int* data,int len){

	printf("# print int array\n");
	for(int i=0;i<len;i++){
		printf(" %d",data[i]);
	}
	printf("\n");

}



