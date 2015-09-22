/*
 * DataProcess.cpp
 *
 *  Created on: Dec 23, 2013
 *      Author: zhoujingbo
 */

#include "DataProcess.h"
#include <limits.h>
#include <algorithm>

DataProcess::DataProcess() {
	// TODO Auto-generated constructor stub

	max = numeric_limits<double>::min();
	min = numeric_limits<double>::max();

}

DataProcess::~DataProcess() {
	// TODO Auto-generated destructor stub
}

vector<string> DataProcess::split(string& str, const char* c) {
	char *cstr, *p;
	vector<string> res;
	cstr = new char[str.size() + 1];
	strcpy(cstr, str.c_str());
	p = strtok(cstr, c);
	while (p != NULL) {
		res.push_back(p);
		p = strtok(NULL, c);
	}
	delete[] cstr;
	return res;
}

string DataProcess::eraseSpace(string origin) {
	int start = 0;
	while (origin[start] == ' ')
		start++;
	int end = origin.length() - 1;
	while (origin[end] == ' ')
		end--;
	return origin.substr(start, end - start + 1);
}

template<typename F>
/**
 * _data:output result
 */
void DataProcess::ReadFile(const char* fname, int fcol,
		F (*atoX)(const char*), vector<F>& _data) {


	string line;
	ifstream ifile(fname);

	_data.clear();



	if (ifile.is_open()) {
		while (getline(ifile, line)) {
			std::size_t found = line.find("#");
			if(found!=std::string::npos) continue;

			vector<string> nstring = split(line, ",");
			string myString = eraseSpace(nstring[fcol]);
			//cout<<"my string"<<myString<<endl;
			F value = (*atoX)(myString.c_str());
			if (value < min&&value>=0)
				min = value;
			if (value > max)
				max = value;

			//if the missing value is -1, ignore it
			//if ( value>=0) {
				_data.push_back(value);

			//}
		}
	}

	ifile.close();


}


/**
 * direct read the time series data, without preprocessing, and the data type is integer
 */
void DataProcess::ReadFileInt(const char * fname, int fcol,vector<int>& _data) {

	ReadFile(fname, fcol, &atoi, _data);

}

void DataProcess::ReadFileDouble(const char * fname, int fcol, vector<double>& _data) {

	ReadFile(fname, fcol,&atof,_data);

}

void printVectorFloat(vector<float> fv){
	for(int i=0;i<fv.size();i++){
		cout<<" "<<fv[i];
	}
	cout<<endl;
}

void DataProcess::ReadFileFloat(const char* fname,int fcol, vector<float>& _data){

	vector<double> df;


	ReadFileDouble(fname,fcol,df);


	_data.resize(df.size());
	for(int i=0;i<df.size();i++){
		_data[i] = (float)(df[i]);
	}

}


template<typename E>
/**
 * _data: output result
 */
void DataProcess::ReadFile(const char * fname,
		E (*atoX)(const char *), vector<vector<E> >& _data) {

	//vector<vector<E>*>* data = new vector<vector<E>*>;
	string line;
	ifstream ifile(fname);

	_data.clear();

	if (ifile.is_open()) {

		while (getline(ifile, line)) {
			std::size_t found = line.find("#");
			if(found!=std::string::npos) continue;
			vector<string> nstring = split(line, ", ");
			vector<E> lv;

			for (int j = 0; j < nstring.size(); j++) {
				E lvi = (*atoX)(nstring[j].c_str());
				lv.push_back(lvi);
			}

			_data.push_back(lv);
		}
	}

	ifile.close();
	//return data;
}


template<typename E>
/**
 * _data: output result
 */
void DataProcess::ReadFile_byComma(const char * fname,
		E (*atoX)(const char *), vector<vector<E> >& _data) {

	//vector<vector<E>*>* data = new vector<vector<E>*>;
	string line;
	ifstream ifile(fname);

	_data.clear();

	if (ifile.is_open()) {

		while (getline(ifile, line)) {
			std::size_t found = line.find("#");
			if(found!=std::string::npos) continue;
			vector<string> nstring = split(line, ",");
			vector<E> lv;

			for (int j = 0; j < nstring.size(); j++) {
				E lvi = (*atoX)(nstring[j].c_str());
				lv.push_back(lvi);
			}

			_data.push_back(lv);
		}
	}

	ifile.close();
	//return data;
}





/**
 * _data:output result
 */
void DataProcess::ReadFileInt(const char * fname, vector<vector<int> >& _data) {
	ReadFile(fname, &atoi,_data);
}


/**
 * _data:output result
 */
void DataProcess::ReadFileFloat(const char* fname, vector<vector<float> >& _data){

	vector<vector<double> > data_db;

	ReadFileDouble(fname, data_db);

	_data.resize(data_db.size());
	for(int i=0;i<data_db.size();i++){
		_data[i].resize(data_db[i].size());
		for(int j=0;j<data_db[i].size();j++){
			_data[i][j] = (float) data_db[i][j];
		}
	}
}

void DataProcess::ReadFileFloat_byCol(const char* fname, vector<vector<float> >& _data){

	vector<vector<double> > data_db;
	ReadFileDouble_byComma(fname, data_db);

	//tranpose the data, one vector for one column
	_data.clear();
	_data.resize(data_db[0].size());
	for(int i=0;i<_data.size();i++){
		_data[i].resize(data_db.size());
	}

	for(int i=0;i<data_db.size();i++){
		for(int j=0;j<data_db[i].size();j++){
			_data[j][i]=(float) data_db[i][j];
		}
	}
}

/**
 * _data:output result
 */
void DataProcess::ReadFileDouble(const char* fname, vector<vector<double> >& _data){
	ReadFile(fname, &atof,_data);
}

/**
 * _data:output result
 */
void DataProcess::ReadFileDouble_byComma(const char* fname, vector<vector<double> >& _data){
	ReadFile_byComma(fname, &atof,_data);
}





/**
 *
 */
void DataProcess::ReadFileWidBucket(const char * fname, int fcol,
		double dw, vector<int>& _dbRes) {

	vector<double> data;
	ReadFileDouble(fname, fcol, data);
	Bucketized(data, dw, _dbRes);

}

void DataProcess::ReadFileBucket(const char * fname, int fcol,
		int bukNum, vector<int> _dbRes) {

	vector<double> data;
	ReadFileDouble(fname, fcol,data);

	double dw = getBukWidth(bukNum);

	Bucketized(data, dw,_dbRes);

}

void DataProcess::Bucketized(vector<double>& data, double dw, vector<int>& _dbRes) {

	_dbRes.clear();
	_dbRes.resize(data.size());
	for (uint i = 0; i < data.size(); i++) {
		int wi = rangPartition(data.at(i), dw);
		_dbRes[i] = wi;

	}

}

double DataProcess::getBukWidth(int bukNum) {

	double dw = (max - min) / (bukNum - 1);
	return dw;

}

int DataProcess::rangPartition(double v, double dw) {

	int vi = (int) (v / dw);
	return vi;
}

template<typename G>
void DataProcess::BuildTrainData(const char * outfile,vector<G>& data, int dim){

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for(int i=0;i<data.size()-dim;i++){
		outf << data[i];
		for (int j = i + 1; j < i + dim + 1; j++)
			outf << "," << data[j];
		outf << endl;

	}
	outf.flush();
	outf.close();

}

template<typename G>
void DataProcess::buildTrainData(vector<G>& data,vector<vector<G> >& data_trn,int dim, int y_offset){

	data_trn.clear();

	for(int i=0;i<data.size()-dim-y_offset;i++){
		vector<G> item;
		item.reserve(dim+1);
		for(int j=i;j<i+dim;j++){
			item.push_back(data[j]);
		}
		item.push_back(data[i+dim+y_offset-1]);

		data_trn.push_back(item);
	}

}

void DataProcess::buildTrainDataFlt(vector<float>& data, vector<vector<float> >& data_trn,int dim, int y_offset){
	buildTrainData(data, data_trn, dim,  y_offset);
}

void DataProcess::BuildTrainDataInt(const char * outfile,vector<int>& data, int dim){
	BuildTrainData(outfile,data,dim);

}

/**
 *
 *
 * keyLow: the nuber of bits for value, the rest of the bits are left for dimensions.
 * In this method, we write file with the bit shift compared with method writeQueryFile(const char* outfile, map<uint,vector<int>* > &query)
 */
void DataProcess::writeInvListQueryFile(const char * outfile,
		map<uint, vector<int> > &query, int keyLow) {
	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<int> >::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<int> v = it->second;

		for (int i = 0; i < v.size(); i++) {
			int vit = v.at(i);
			int d = (i << keyLow) + vit; //make the composite key
			outf << d << " ";
		}

		outf << endl;
	}

	outf.close();
}



void DataProcess::writeQueryFile_float(const char* outfile,
		map<uint, vector<float> > &query){

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<float> >::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<float> v = it->second;

		for (int i = 0; i < v.size(); i++) {
			float vit = v.at(i);
			outf << vit << " ";
			//cout<< vit << " ";
		}
		outf << endl;
		// cout<<endl;
	}

	outf.close();
}



void DataProcess::writeQueryFile_int(const char* outfile,
		map<uint, vector<int> > &query){

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<int> >::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<int> v = it->second;

		for (int i = 0; i < v.size(); i++) {
			float vit = v.at(i);
			outf << vit << " ";
			//cout<< vit << " ";
		}
		outf << endl;
		// cout<<endl;
	}

	outf.close();
}


/**
 * in this method, we do not write the file with bit shift,
 * compared with writeInvListQueryFile(const char * outfile,map<uint,vector<int>* > &query, int keyLow)
 *//*
template<class H>
void DataProcess::writeQueryFile(const char* outfile,
		map<uint, vector<H> > &query) {

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<H> >::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<H> v = it->second;

		for (int i = 0; i < v.size(); i++) {
			H vit = v.at(i);
			outf << vit << " ";
			//cout<< vit << " ";
		}
		outf << endl;
		// cout<<endl;
	}

	outf.close();

}
*/
void DataProcess::writeBinaryQueryFile(const char* outfile,
		map<uint, vector<int>*> &query){
	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	for (map<uint, vector<int>*>::iterator it = query.begin();
			it != query.end(); ++it) {
		vector<int> *v = it->second;

		for (int i = 0; i < v->size(); i++) {
			int vit =  v->at(i);
			outf.write((char*) &vit, sizeof(vit));
		}
		outf << endl;
		// cout<<endl;
	}
	outf.close();

}



void DataProcess::writeBinaryFile(const char * outfile, int numOfDim, int maxValuePerDim,
		 map<uint, vector<int> >& im) {

	ofstream outf;
	outf.open(outfile, ios::binary | ios::out);

	int countK = 0;
	int countE = 0;

	//write number of dimension and number of features

	outf.write((char*) &numOfDim, sizeof(uint));
	outf.write((char*) &maxValuePerDim, sizeof(uint));

	for (map<uint, vector<int> >::iterator it = im.begin(); it != im.end();
			++it) {
		uint key = it->first;
		outf.write((char*) &key, sizeof(uint));
		countK++;

		vector<int> v = it->second;
		uint s = v.size();

		outf.write((char*) &s, sizeof(s));

		for (int vi = 0; vi < v.size(); vi++) {
			countE++;
			int vit = (v)[vi];
			outf.write((char*) &vit, sizeof(vit));
		}
	}

	cout << "The total number of keys in idx is:" << countK << endl;
	cout << "The total number of elements in idx is:" << countE << endl;
	outf.close();

}


void DataProcess::z_normalizeData(string inputFilename, int fcol_start,int fcol_end, string outputFilename,double rangeMultipler){

	vector<double> mean(fcol_end-fcol_start+1,0);
	vector<double> dev(fcol_end-fcol_start+1,0);

	for(int i=fcol_start;i<=fcol_end;i++){
		vector<double> data;
		ReadFileDouble(inputFilename.c_str(), i,data);
		z_normalizeData_perVector(data, mean[i-fcol_start],dev[i-fcol_start]);
	}

	string line;
	ifstream ifile(inputFilename.c_str());

	ofstream outf;
	outf.open(outputFilename.c_str(),  ios::out);

	if (ifile.is_open()) {
		while (getline(ifile, line)) {
			std::size_t found = line.find("#");
			if (found != std::string::npos) {
				outf << line << endl;
				continue;
			}

			vector<string> nstring = split(line, ",");

			for (int i = 0; i < nstring.size(); i++) {
				if (i < fcol_start||i>fcol_end) {
					outf << nstring[i] << ",";
				} else {
					string myString = eraseSpace(nstring[i]);
					double value = (atof)(myString.c_str());
					outf << (value - mean[i-fcol_start]) / dev[i-fcol_start] * rangeMultipler << ",";
				}
			}
			outf << endl;
		}
	}




}

void DataProcess::z_normalizeData_perVector(vector<double>& data, double& mean, double& dev){

	double sum = 0, sumSqau = 0;

	for (size_t i = 0; i < data.size(); i++) {
		sum += data[i];
		sumSqau += (data[i] * data[i]);
	}

	mean = sum / data.size();
	dev = sqrt(sumSqau / data.size() - mean * mean);



}

void DataProcess::z_normalizeData(string inputFilename,  int fcol, string outputFilename, double rangeMultipler){

	vector<double> data;
	data.clear();
	cout<<"start processing file:"<<inputFilename<<" col:"<<fcol<<endl;
	ReadFileDouble(inputFilename.c_str(), fcol, data);


	double sum=0,sumSqau=0;

	for(size_t i=0;i<data.size();i++){
		sum+=data[i];
		sumSqau+=(data[i]*data[i]);
	}

	double mean = sum/data.size();
	double dev = sqrt(sumSqau/data.size() - mean*mean);

	string line;
	ifstream ifile(inputFilename.c_str());

	ofstream outf;
	outf.open(outputFilename.c_str(),  ios::out);
	outf<<"#mean:"<<mean<<" standard variance:"<<dev<<endl;


	if (ifile.is_open()) {
			while (getline(ifile, line)) {
				std::size_t found = line.find("#");
				if(found!=std::string::npos) {
				outf<<line<<endl;
				continue;
				}

				vector<string> nstring = split(line, ",");
				string myString = eraseSpace(nstring[fcol]);
				double value = (atof)(myString.c_str());
				//cout<<"my string"<<myString<<endl;
				for(int i=0;i<nstring.size();i++){
					if(i!=fcol){
						outf<<nstring[i]<<",";
					}else{
						outf<< (value-mean)/dev*rangeMultipler<<",";
					}
				}
				outf<<endl;
			}
	}

	cout<<"output normalized data in:"<<outputFilename<<endl;
	outf.close();

}



//special function for processing Dodgers
void DataProcess::remove_missData(string inputFilename,  int fcol, string outputFilename){

	string line;
	ifstream ifile(inputFilename.c_str());

	ofstream outf;
	outf.open(outputFilename.c_str(),  ios::out);

	if (ifile.is_open()) {
		while (getline(ifile, line)) {
			std::size_t found = line.find("#");
			if(found!=std::string::npos) continue;

			vector<string> nstring = split(line, ",");
			string myString = eraseSpace(nstring[fcol]);
			//cout<<"my string"<<myString<<endl;
			double value = (atof)(myString.c_str());

			//if the missing value is -1, ignore it
			if ( value==-1) {
				line.insert(0,"#");
			}
			outf<<line<<endl;
		}
	}

	ifile.close();
	outf.close();
}

void DataProcess::run() {

}

