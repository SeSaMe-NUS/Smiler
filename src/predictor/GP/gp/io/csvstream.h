#ifndef CSVSTREAM_H_
#define CSVSTREAM_H_

// Uncomment following line to enable debugging messages
// #define DEBUG
#include <iostream>
#include <cstring>
#include <vector>
#include <cassert>

//#include <itpp/itbase.h>
//using namespace itpp;//comment by jignbo


#include "armadillo"//add by jingbo

using namespace std;
using namespace arma;

#define debug_msg(msg)

/**
 * This class provides support for reading from and
 * writing to CSV (comma separated values) files. At the moment,
 * it only supports reading to and writing from matrices, and does
 * not (yet) work like a proper C++ stream, although support for 
 * streamed input/output will be added later.
 */
class csvstream
{
public:
	csvstream();
	virtual ~csvstream();
	
	int read(mat &matrix, const string filename);
	int write(const mat matrix, const string filename, int decimals = 5);

};

#endif /*CSVSTREAM_H_*/
