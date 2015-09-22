/***************************************************************************
 *   AstonGeostats, algorithms for low-rank geostatistical models          *
 *                                                                         *
 *   Copyright (C) Ben Ingram, 2008-2009                                   *
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
 //
 // Scaled Conjugate Gradient algorithm based heavily on the Netlab
 // toolbox originally written in Matlab by Prof Ian Nabney
 //
 // History:
 //
 // 12 Dec 2008 - BRI - First implementation. 
 //

#include "SCGModelTrainer.h"

SCGModelTrainer::SCGModelTrainer(Optimisable& m) : ModelTrainer(m)
{
	algorithmName = "Scaled Conjugate Gradient";
}

SCGModelTrainer::~SCGModelTrainer()
{

}

/**
 * TODO:
 *
 * find the parameter values to minimize the objective function
 */
void SCGModelTrainer::Train(int numIterations)
{
	double sigma0 = 1.0e-4;
	double beta = 1.0;
	double betaMin = 1.0e-15;
	double betaMax = 1.0e100;
	double kappa = 0.0;
	double mu = 0.0;	
	double sigma;
	double theta = 0.0;
	double delta; // check delta and Delta
	double alpha, fOld, fNew, fNow;
	double Delta;
    double EPS = arma::math::eps();
    
	vec direction, gradNew, gradOld, gPlus, xPlus, xNew;
	vec x = getParameters();
				
	int numParams, numSuccess;
	bool success;
		
	fOld = errorFunction(x);
	fNow = fOld;

	gradNew = errorGradients(x);
	gradOld = gradNew;

	direction = -gradNew;

	numParams = gradNew.size();
	success = true;
	numSuccess = 0;
		
	if(gradientCheck)
	{
		checkGradient();
	}
		
	// Main loop
	for (int j = 1; j <= numIterations; j++ )
	{
		if (success)
		{
			mu = dot(direction, gradNew);
			if ( mu >= 0.0 )
			{
				direction = -gradNew;
				mu = dot(direction, gradNew);
			}
				
			kappa = dot(direction, direction);

			// eps exists in ITPP? remember to check this!
			if(kappa < EPS)
			{
				functionValue = fNow;
				setParameters(x);
				return;
			}
			sigma = sigma0 / sqrt(kappa);
			xPlus = x + (sigma * direction);
			gPlus = errorGradients(xPlus);
			theta = dot(direction, gPlus - gradNew) / sigma;	
		}
		
		delta = theta + (beta * kappa);
			
		if ( delta <= 0.0 )
		{	
		    delta = beta * kappa;
			beta = beta - ( theta / kappa );
		    //double olddelta = delta;
		    //double oldbeta = beta;
		    //beta = 2.0*(oldbeta - olddelta/kappa);
		    //delta = oldbeta*kappa - olddelta;
		}
		alpha = - ( mu / delta );
		
		xNew = x + (alpha * direction);
		fNew = errorFunction(xNew);

		Delta = 2.0 * ( fNew - fOld ) / (alpha * mu);
		if ( Delta >= 0.0 )
		{
			success = true;
			numSuccess++;
			x = xNew;
			
			// RB: Do we need to set parameters here?
			setParameters(x);
			fNow = fNew;
		}
		else
		{
			success = false;
			fNow = fOld;
		}
			
		if(display)
		{
			Rprintf("Cycle %d   Error %f  Scale %f\n", j, fNow, beta);
		}

		if(success)
		{
			if ((max(alpha * direction) < parameterTolerance) && (abs( fNew - fOld )) < errorTolerance )
			{
				functionValue = fNew;
				// setParameters(x); 
				return;
			}
			else
			{
				fOld = fNew;
				gradOld = gradNew;
				gradNew = errorGradients(x);
				
				if(dot(gradNew, gradNew) < 1e-16)
				{
					functionValue = fNew;
					// setParameters(x);
					return;
				}
			}
		}
			
		if ( Delta < 0.25 )
		{
			beta = min( 4.0 * beta, betaMax );
		}
			
		if ( Delta > 0.75 )
		{
			beta = max( 0.5 * beta, betaMin );
		}
			
		if ( numSuccess == numParams )
		{
			direction = -gradNew;
			numSuccess = 0;
		}
		else
		{
			if (success)
			{
				double gamma = dot(gradOld - gradNew, (gradNew / mu));
				direction = (gamma * direction) - gradNew;
			}
		}
	}
		
	if(display)
	{
		Rprintf("Warning: Maximum number of iterations has been exceeded\n");
	}

	functionValue = fOld;
	// setParameters(x);
	
	// Check last gradient (to make sure everything went fine
	if (gradientCheck) checkGradient();
	
	return;
}


