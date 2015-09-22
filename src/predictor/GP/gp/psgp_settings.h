/*
 * psgp_settings.h
 *
 *  Created on: 21 Jan 2012
 *      Author: barillrl
 */

#ifndef PSGP_SETTINGS_H_
#define PSGP_SETTINGS_H_

//-----------------------------------------------------------------------------
// CONSTANTS AND OTHER GENERAL PARAMETERS
//-----------------------------------------------------------------------------

// Max number of parameters for PSGP (this limit is set by the R code)
#define NUM_PSGP_PARAMETERS 16

// Maximum number of observations kept for param estimation
#define MAX_OBS 1000

// Maximum number of active points
#define MAX_ACTIVE_POINTS 400

// Likelihood to nugget ratio
#define LIKELIHOOD_NUGGET_RATIO 0.01

// Number of sweeps through data with changing/fixed active set
#define NUM_SWEEPS_CHANGING 1
#define NUM_SWEEPS_FIXED 1

// Whether to use a GP instead of PSGP for parameter estimation
#define PARAMETER_ESTIMATION_USING_GP false

// Outer loops in parameter estimation for PSGP
#define PSGP_PARAM_ITERATIONS 3

// Inner loop (i.e. SCG iterations in each outer loop) for PSGP
#define PSGP_SCG_ITERATIONS 5

// Define whether to use full prediction (all data at once) or
// split prediction (by chunks of data)
#define USING_CHUNK_PREDICTION true

// Size of prediction chunk (in number of observations)
#define CHUNK_SIZE 1000


#endif /* PSGP_SETTINGS_H_ */
