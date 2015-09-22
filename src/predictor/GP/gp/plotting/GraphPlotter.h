//////////////////////////////////////////////////////////////////////////////
//
// GraphPlotter :-
//   provides a simple interface to GNUPlot from C++.  Uses IT++ data types
//   for representing data to be plotted 
//
// (c) Ben Ingram, 2009
// e-mail: ingrambr@gmail.com
//
//////////////////////////////////////////////////////////////////////////////


/**
 * RB: Note that for the plotter to work as expected, it is necessary to pause
 * the program before termination (e.g. using getchar()). The data to be plotted
 * is stored in temporary files, which get deleted on program termination (when
 * the GraphPlotter is destroyed). This can occur before the gnuplot thread
 * displays the data, and can result in errors (if the data has been deleted before
 * begin read by gnuplot).
 **/

#ifndef GRAPHPLOTTER_H_
#define GRAPHPLOTTER_H_

#include <stdarg.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

//#include "itpp/itbase.h"//comment by jingbo
//using namespace itpp;//comment by jingbo

#include "armadillo"//add by jingbo

using namespace std;
using namespace arma;


enum pointStyle {LINE = 99, DOT = 0, CROSS = 1, XCROSS = 2, STAR = 3, SQUARE = 4, SQUAREFILLED = 5, CIRCLE = 6, CIRCLEFILLED = 7, TRIANGLE = 8, TRIANGLEFILLED = 9, UTRIANGLE = 10, UTRIANGLEFILLED = 11, DIAMOND = 12, DIAMONDFILLED = 13};
enum plotColor {BLACK = -1, RED = 1, GREEN = 2, BLUE = 3, PURPLE = 4, CYAN = 5, YELLOW = 6};

class GraphPlotter
{
public:
	GraphPlotter();              
	~GraphPlotter();

	void sendCommand(const char* command, ...);
	void setYLabel(const string label);
	void setXLabel(const string label);
	void plotPoints(const vec &x, const vec &y, const string title, const pointStyle ps, const plotColor pc);
	void clearPlot();

private:
	FILE *plotterStream;
	vector<string> tempList;      
};

#endif
