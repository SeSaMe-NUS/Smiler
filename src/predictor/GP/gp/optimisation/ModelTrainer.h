/***************************************************************************
 *   AstonGeostats, algorithms for low-rank geostatistical models          *
 *                                                                         *
 *   Copyright (C) Ben Ingram, 2008                                        *
 *                                                                         *
 *   Ben Ingram, IngramBR@Aston.ac.uk                                      *
 *   Neural Computing Research Group,                                      *
 *   Aston University,                                                     *
 *   Aston Street, Aston Triangle,                                         *
 *   Birmingham. B4 7ET.                                                   *
 *   United Kingdom                                                        *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef MODELTRAINER_H_
#define MODELTRAINER_H_

#include "Optimisable.h"

#include "../psgp_common.h"
#include "armadillo"
#include <string>

//using namespace arma;//add by jingbo

const double PHI = ((1.0 + sqrt(5.0)) / 2.0);
const double CPHI = (1.0 - (1.0/PHI));
const double TOL = sqrt( arma::math::eps() );
const double TINY = (1.0e-10);

using namespace psgp_arma;

class ModelTrainer
{
public:
	ModelTrainer(Optimisable& m);
	virtual ~ModelTrainer();

	void Summary() const;
	virtual void Train(int numIterations) = 0;

	void setDisplay(bool b) {display = b;};
	void setCheckGradient(bool b) {gradientCheck = b;};
	void setAnalyticGradients(bool b) {analyticGradients = b;};
	void setErrorTolerance(double d) {errorTolerance = d;};

	void setLineMinimiserIterations(int i) {lineMinimiserIterations = i;};
	void setLineMinimiserParameterTolerance(double d) {lineMinimiserParameterTolerance = d;};

	void setFiniteDifferenceDelta(double d) {epsilon = d;};

	void setOptimisationMask(uvec& m) {optimisationMask = m; maskSet = true;};
	
	void checkGradient();
	
protected:
	
	vec numericalGradients(const vec params);
	double calculateNumericalGradient(const int parameterNumber, const vec params);

	double errorFunction(vec params);
	vec errorGradients(vec params);

	vec getParameters();
	void setParameters(const vec p);

	double lineFunction(vec x, double lambda, vec direction);
	void lineMinimiser(double &fx, double &x, double fa, vec params, vec direction);
	void bracketMinimum(double &br_min, double &br_mid, double &br_max, double a, double b, double fa, vec params, vec direction);

	Optimisable& model;

	bool display;
	double errorTolerance;
	double parameterTolerance;
	bool gradientCheck;
	bool analyticGradients;

	int functionEvaluations;
	int gradientEvaluations;
	double functionValue;

	int lineMinimiserIterations;
	double lineMinimiserParameterTolerance;

	bool maskSet;
	uvec optimisationMask;

	double epsilon;

	string algorithmName;
};


#endif /*MODELTRAINER_H_*/
