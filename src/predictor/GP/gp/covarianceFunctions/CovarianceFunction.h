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

#ifndef COVARIANCEFUNCTION_H_
#define COVARIANCEFUNCTION_H_



#include <string>
#include "../psgp_common.h"

#include "Transform.h"
#include "IdentityTransform.h"
#include "LogTransform.h"

using namespace arma;

class CovarianceFunction
{


public:
	CovarianceFunction(string name);
	virtual ~CovarianceFunction();
	
	virtual void computeSymmetric(double& c, const vec& X) const;
	virtual void computeSymmetric(mat& C, const mat& X) const;
	
	virtual void computeSymmetricGrad(vec& V, const mat& X) const;
	
	virtual void computeCovariance(vec& C, const mat& X, const vec& x) const;
	virtual void computeCovariance(mat& C, const mat& X1, const mat& X2) const;
	
	virtual void computeDiagonal(mat& C, const mat& X) const;
	virtual void computeDiagonal(vec& C, const mat& X) const;	
	
	virtual double computeElement(const vec& A, const vec& B) const = 0;
	virtual double computeDiagonalElement(const vec& A) const = 0;
	
	virtual void getParameterPartialDerivative(mat& PD, const unsigned int parameterNumber, const mat& X) const = 0;
	

	virtual void   setParameter(const unsigned int parameterNumber, const double value) = 0;
	virtual double getParameter(const unsigned int parameterNumber) const = 0;
	
	virtual string getParameterName(const unsigned int parameterNumber) const = 0;
	virtual vec getLenthScaleVector() const {return NULL;}
	virtual vec getVarScaleVector() const {return NULL;}


// add something about transformations here

	virtual void setTransform(unsigned int parameterNumber, Transform* newTransform);
	virtual Transform* getTransform(unsigned int parameterNumber) const;
	
	virtual void setParameters(const vec p);
	virtual vec getParameters() const;
	
	unsigned int getNumberParameters() const;
	
	virtual void displayCovarianceParameters(int nspaces = 0) const;

	void computeDistanceMatrix(mat& DM, const mat& X) const;
	void computeDistanceMatrix(uint dim_i, mat& DM, const mat& X) const;

	void computeDistanceVector(uint dim_i,vec& DV, const mat& X, const vec& Xseed) const;

protected:
	virtual void setDefaultTransforms();
	
	
	string covarianceName;
	unsigned int numberParameters;
	bool transformsApplied;

private:
	vector<Transform *> transforms;



};

#endif /*COVARIANCEFUNCTION_H_*/
