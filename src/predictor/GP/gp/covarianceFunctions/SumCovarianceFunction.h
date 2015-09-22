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

#ifndef SUMCOVARIANCEFUNCTION_H_
#define SUMCOVARIANCEFUNCTION_H_

#include "CovarianceFunction.h"
#include "Transform.h"

#include <cmath>
#include "../psgp_common.h"

class SumCovarianceFunction : public CovarianceFunction
{
public:
	// SumCovarianceFunction(vector<CovarianceFunction> cfVec);
	SumCovarianceFunction();//add by jingbo
	SumCovarianceFunction(CovarianceFunction& cf);
	
	virtual ~SumCovarianceFunction();

	double computeElement(const vec& A, const vec& B) const;
	double computeDiagonalElement(const vec& A) const;
	
	void getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const;
	
	void setParameter(const unsigned int parameterNumber, const double value) const;
	double getParameter(const unsigned int parameterNumber) const;

	// C++ 11 - no need to declare the implementation
	void getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X);
	
	void setParameter(const unsigned int parameterNumber, const double value);
	double getParameter(const unsigned int parameterNumber);
	
	string getParameterName(const unsigned int parameterNumber);
	
	string getParameterName(const unsigned int parameterNumber) const;
	
	void setParameters(const vec p);
	vec getParameters() const;

	void setTransform(unsigned int parameterNumber, Transform* newTransform);
	Transform* getTransform(unsigned int parameterNumber) const;

	void addCovarianceFunction(CovarianceFunction& cf);
	void displayCovarianceParameters(int nspaces = 0) const;
	
private:
	vector<CovarianceFunction *> covFunctions;
};

#endif /*SUMCOVARIANCEFUNCTION_H_*/
