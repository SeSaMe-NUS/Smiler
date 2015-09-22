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
 // ModelTrainer abstract class.  Includes implementation of line minimiser
 // and minimum bracket algorithms which are based heavily on the Netlab
 // toolbox originally written in Matlab by Prof Ian Nabney
 //
 // History:
 //
 // 12 Dec 2008 - BRI - First implementation. 
 //
 
#include "ModelTrainer.h"

ModelTrainer::ModelTrainer(Optimisable& m) : model(m)
{
	// default training options
	display = true;
	errorTolerance = 1.0e-6;
	parameterTolerance = 1.0e-4;
	
	gradientCheck = false; // perform a gradient check
	analyticGradients = true; // use analytic gradient calculations
	
	functionEvaluations = 0; // reset function counter
	gradientEvaluations = 0; // reset gradient counter
	functionValue = 0.0; // current function value
	
	lineMinimiserIterations = 10;
	lineMinimiserParameterTolerance = 1.0e-4;
	
	maskSet = false; // do we want to mask certain parameters
	
	epsilon = 1.0e-6;	// delta for finite difference approximation
}

ModelTrainer::~ModelTrainer()
{
}


double ModelTrainer::errorFunction(vec params)
{
	functionEvaluations++;
	vec xOld = getParameters();
	
	// Compute error
	setParameters(params);
	double err = model.objective();
	
	// Don't forget to reset parameters to their initial state
	setParameters(xOld);
	
	return err;
}

vec ModelTrainer::errorGradients(vec params)
{
    vec xOld = getParameters();
    vec grad;

    
	if(analyticGradients)
	{
		gradientEvaluations++;
		
		// Computer error gradient
		setParameters(params);
		grad = model.gradient();
		
		// Don't forget to reset parameters to their initial state
		setParameters(xOld);
	}
	else
	{
		grad = numericalGradients(params);
	}
	
	// Gradient is zero for parameters not in optimisation mask
	if (maskSet) {
	    for (unsigned int i=0; i<optimisationMask.n_elem; i++) {
	        if (optimisationMask(i) == 0) grad(i) = 0.0;
	    }
	}
	
	return grad;
}

vec ModelTrainer::numericalGradients(const vec params)
{
	vec g;
	int numParams = params.size();

	g.set_size(numParams);
	for(int i=0; i < numParams; i++)
	{
		g(i) = calculateNumericalGradient(i, params);
	}
	return(g);
}

double ModelTrainer::calculateNumericalGradient(const int parameterNumber, const vec params)
{
	vec xNew;
	double fplus, fminus;
	
	xNew = params;
	xNew(parameterNumber) = xNew(parameterNumber) + epsilon;
		
	fplus = errorFunction(xNew);
		
	xNew = params;
	xNew(parameterNumber) = xNew(parameterNumber) - epsilon;
	fminus = errorFunction(xNew);
	
	return (0.5 * ((fplus - fminus) / epsilon));
}
	
void ModelTrainer::setParameters(const vec p)
{
	if(maskSet)
	{
		vec maskedParameters = model.getParametersVector();
		for(unsigned int i = 0 ; i < optimisationMask.size() ; i++)
		{
			if(optimisationMask(i)==1) maskedParameters(i) = p(i);
		}
		model.setParametersVector(maskedParameters);
	}
	else
	{
		model.setParametersVector(p);
	}
}

vec ModelTrainer::getParameters()
{
    // It is probably safer to return all parameters and simply set 
    // the gradient of the non-masked ones to zero, rather than extracting
    // subsets of the parameters.
    /*
    if(maskSet)
	{
		vec p = model.getParametersVector();
		vec maskedParameters;
		
		for(int i = 0 ; i < optimisationMask.size() ; i++)
		{
			if(optimisationMask(i) == 1)
			{
				maskedParameters = concat(maskedParameters, p(i));
			}
		}
		return maskedParameters;
	}
	else
	*/
	{
		return model.getParametersVector();
	}
}

void ModelTrainer::checkGradient()
{
	vec xOld = getParameters();
	vec gNew = model.gradient();
	int numParams = gNew.size();
	
	double delta;

	Rprintf ("==========================\n");
	Rprintf("Gradient check\n");
	Rprintf("Delta, Analytic, Diff\n");
	Rprintf("--------------------------\n");;

	for(int i=0; i < numParams; i++)
	{
		if(maskSet)
		{
			if(optimisationMask(i)==1)
			{
				delta = calculateNumericalGradient(i, xOld);
			}
			else
			{
				delta = 0.0;
				gNew(i) = 0.0;
			}
		}
		else
		{
			delta = calculateNumericalGradient(i, xOld);
		}

		Rprintf(" %f %f %f\n", delta , gNew(i) , abs(delta - gNew(i)));

	}
	Rprintf("==========================\n");

}

void ModelTrainer::Summary() const
{
	Rprintf("================================================\n");
	Rprintf("Training summary     : %s\n", algorithmName.c_str());
	Rprintf("------------------------------------------------\n");
	Rprintf("Error tolerance      : %f", errorTolerance );
	Rprintf("Parameter tolerance  : %f", parameterTolerance );
	Rprintf("Function evaluations : %d", functionEvaluations );
	Rprintf("Gradient evaluations : %d", gradientEvaluations );
	Rprintf("Function value       : %f\n", functionValue);
	Rprintf("================================================\n");;
}

double ModelTrainer::lineFunction(vec param, double lambda, vec direction)
{
	double f;
	vec xOld = getParameters();
	vec v = lambda * direction + param;
	f = errorFunction( v );
	setParameters(xOld);
	return(f);
}

void ModelTrainer::lineMinimiser(double &fx, double &x, double fa, vec params, vec direction)
{
	double br_min, br_mid, br_max;
	double w, v, e, d = 0, r, q, p, u, fw, fu, fv, xm, tol1;

	bracketMinimum(br_min, br_mid, br_max, 0.0, 1.0, fa, params, direction);

	w = br_mid;
	v = br_mid;
	x = v;
	e = 0.0;
	fx = lineFunction(params, x, direction);
	fv = fx;
	fw = fx;

	for(int n = 1; n <= lineMinimiserIterations; n++)
	{
		xm = 0.5 * (br_min + br_max);
		tol1 = TOL * abs(x) + TINY;
 
		if((abs(x - xm) <= lineMinimiserParameterTolerance) & ((br_max - br_min) < (4 * lineMinimiserParameterTolerance)))
		{
			functionValue = fx;
			return;
		}
		
		if (abs(e) > tol1)
		{
			r = (fx - fv) * (x - w);
			q = (fx - fw) * (x - v);
			p = ((x - v) * q) - ((x - w) * r);
			q = 2.0 * (q - r);

			if (q > 0.0)
			{
				p = -p;
			}
			
			q = abs(q);

			if ((abs(p) >= abs(0.5 * q * e)) | (p <= (q * (br_min - x))) | (p >= (q * (br_max - x))))
			{
				if (x >= xm)
				{
					e = br_min - x;
				}
				else
				{
					e = br_max - x;
				}
		      d = CPHI * e;
		   }
			else
			{
				e = d;
				d = p / q;
				u = x + d;
				if (((u - br_min) < (2 * tol1)) | ((br_max - u) < (2 * tol1)))
				{
					d = sign(xm - x) * tol1;
				}
			}
		}
		else
		{
			if (x >= xm)
			{
				e = br_min - x;
			}
			else
			{
				e = br_max - x;
			}
			d = CPHI * e;
		}
	
		if (abs(d) >= tol1)
		{
			u = x + d;
		}
		else
		{
			u = x + sign(d) * tol1;
		}
		
		fu = lineFunction(params, u, direction);

		if (fu <= fx)
		{
			if (u >= x)
			{
				br_min = x;
			}
			else
			{
				br_max = x;
			}
			v = w; w = x;x = u;
			fv = fw; fw = fx;	fx = fu;
		}
		else
		{
			if (u < x)
			{
				br_min = u;
			}   
			else
			{
				br_max = u;
			}
		
			if ((fu <= fw) | (w == x))
			{
				v = w; w = u;
				fv = fw; fw = fu;
			}
			else
			{
				if ((fu <= fv) | (v == x) | (v == w))
				{
					v = u; fv = fu;
				}
			}
		}
	}
}

void ModelTrainer::bracketMinimum(double &br_min, double &br_mid, double &br_max, double a, double b, double fa, vec params, vec direction)
{
	double c, fc, r, q, u, ulimit, fu = 0.0;
	double fb = lineFunction(params, b, direction);

	bool bracket_found;

	const double max_step = 10.0;

	if(fb > fa)
	{
		c = b;
		b = a + (c - a) / PHI;
		fb = lineFunction(params, b, direction);
		
		while (fb > fa)
		{
			c = b;
    		b = a + (c - a) / PHI;
    		fb = lineFunction(params, b, direction);
		}
	}
	else
	{
		c = b + PHI * (b - a);
		fc = lineFunction(params, c, direction);

		bracket_found = false;
  
		while (fb > fc)
		{
			r = (b - a) * (fb - fc);
			q = (b - c) * (fb - fa);
			u = b - ((b - c) * q - (b - a) * r) / (2.0 * (sign(q - r) * max(abs(q - r), TINY)));

			ulimit = b + max_step * (c - b);
    
			if ((b - u) * (u - c) > 0.0)
			{
				fu = lineFunction(params, u, direction);
				if (fu < fc)
				{
					br_min = b;	br_mid = u;	br_max = c;
					return;
				}
 				else
 				{
 					if (fu > fb)
 					{
						br_min = a;	br_mid = c;	br_max = u;
						return;
					}
				}
				u = c + PHI * (c - b);
			}
			else
			{
				if ((c - u) * (u - ulimit) > 0.0)
				{
					fu = lineFunction(params, u, direction);

					if (fu < fc)
					{
						b = c;
						c = u;
						u = c + PHI * (c - b);
					}
					else
					{
						bracket_found = true;
					}
				}
				else
				{
					if ((u - ulimit) * (ulimit - c) >= 0.0)
					{
						u = ulimit;
					}
					else
					{
						u = c + PHI * (c - b);
					}
				}
			}
			
			if(!bracket_found)
			{
				fu = lineFunction(params, u, direction);
			}
			a = b; b = c; c = u;
			fa = fb; fb = fc; fc = fu;
		}
	}
	
	br_mid = b;

	if(a < c)
	{
		br_min = a; br_max = c;
	}
	else
	{
		br_min = c; br_max = a;
	}
}




