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

#ifndef ARDGAUSSIANCF_H_
#define ARDGAUSSIANCF_H_



#include "CovarianceFunction.h"
#include "../psgp_common.h"

#define DELPHIHEADER_NO_IMPLICIT_NAMESPACE_USE

#include <cmath>
#include <cassert>
//#include <itpp/itbase.h>


using namespace arma;


class ARDGaussianCF : public CovarianceFunction
{
public:
	ARDGaussianCF(vec lengthscales, double var);
	ARDGaussianCF(double lengthscale, double var, int dims);
	//ARDGaussianCF(vec parameters);
	
	virtual ~ARDGaussianCF();

	inline double computeElement(const vec& A, const vec& B) const;
	inline double computeDiagonalElement(const vec& A) const;

	void getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const;
	//void getParameterPartialDerivative(vec & pdv, const unsigned int parameterNumber, const mat & X, const vec & Xseed) const;
	vec getLenthScaleVector() const;
	vec getVarScaleVector() const;
	
	void setParameter(unsigned int parameterNumber, const double value);
	double getParameter(unsigned int parameterNumber) const;
	
	string getParameterName(unsigned int parameterNumber) const;
	
private:
	inline double calcGaussian(const vec& V) const;
	inline double calcGaussianDiag() const;
	
	double variance;
	vec ranges;
};

#endif /*ARDGAUSSIANCF_H_*/
