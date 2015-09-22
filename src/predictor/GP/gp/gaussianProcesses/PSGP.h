/***************************************************************************
 *   AstonGeostats, algorithms for low-rank geostatistical models          *
 *                                                                         *
 *   Copyright (C) Remi Barillec, Ben Ingram, 2008-2009                    *
 *                                                                         *
 *   Remi Barillec, r.barillec@aston.ac.uk
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

#ifndef PSGP_H_
#define PSGP_H_

#include "ForwardModel.h"
#include "../optimisation/Optimisable.h"
#include "../covarianceFunctions/CovarianceFunction.h"
#include "../likelihoodModels/LikelihoodType.h"

#include <vector>
#include "../psgp_common.h"

#define LAMBDA_TOLERANCE 1e-10

#ifndef SEQUENTIALGP_H_
    enum ScoringMethod { Geometric, MeanComponent, FullKL };
    enum LikelihoodCalculation { FullEvid, Approximate, UpperBound };
#endif

using namespace std;
using namespace arma;

using namespace psgp_arma;

// We went through 3 implementations of the algorithm as we optimised it. 
// Although we could have left the latest version only (V3) as it is the
// fastest/most memory efficient, we have left the previous versions 
// with debug purposes. These flags correspond to the 3 implementations
// of the algorith, V1 being the oldest (and least efficient) and V3
// the newest (and most efficient). V3 is used by default.
enum AlgoVersion { ALGO_V1, ALGO_V2, ALGO_V3 };
    
class PSGP : public ForwardModel, public Optimisable
{
public:
    PSGP(mat& X, vec& Y, CovarianceFunction& cf, int nActivePoints=400, int _iterChanging=1, int _iterFixed=2);
	virtual ~PSGP();

	void computePosterior(const LikelihoodType& noiseModel);
	void computePosterior(const arma::ivec& LikelihoodModel, const std::vector<LikelihoodType *> noiseModels);
	void resetPosterior();
	void recomputePosterior();
	
	void computePosteriorFixedActiveSet(const LikelihoodType& noiseModel, uvec iActive);
	void recomputePosteriorFixedActiveSet(const LikelihoodType& noiseModel);
	
	/**
	 * Make predictions at a set of locations Xpred. The mean and variance
	 * are returned in the Mean and Variance vectors. To use a different  
	 * covariance function to the training one (useful to do non noisy predictions),
	 * you can pass an optional CovarianceFunction object.
	 */  
	void makePredictions(vec& Mean, vec& Variance, const mat& Xpred, CovarianceFunction &cf) const;
	void makePredictions(vec& Mean, vec& Variance, const mat& Xpred) const;
	
	
	void setAlgoVersion(AlgoVersion version) { algoVersion = version; }
	void setGammaTolerance(double gammaMin) { gammaTolerance = gammaMin; }
	
	vec simulate(const mat& Xpred, bool approx) const;

	// Set/Get/Print methods for covariance parameters
	vec  getParametersVector() const;
	void setParametersVector(const vec p);
	void displayModelParameters() const;

	// Optimisation function
	double objective() const;
	vec    gradient() const;
	
	/**
	 * Accessors/Modifiers
	 */
	int  getSizeActiveSet()      { return sizeActiveSet; }
	uvec getActiveSetIndices()   { return idxActiveSet; }
	mat  getActiveSetLocations() { return ActiveSet; }
	vec  getActiveSetObservations() { return Observations.elem(idxActiveSet); }

	void setActiveSetSize(int n) { maxActiveSet = n; }
	void setActiveSet(uvec activeIndexes, mat activeLocations);
	void setLikelihoodType(LikelihoodCalculation lc);
	
	void setDisplay(bool display){this->display=display;}
		
private:

	bool display;

    // Version of the algorithm to use (should be V3, the 
    // latest and fastest) unless an older version is really needed
    // (for backwards comparison/debugging)
    AlgoVersion algoVersion;
    
    // Inputs and outputs
    mat& Locations;
    vec& Observations;
    
    unsigned int nObs;  // Number of observations
    
    // Covariance function
    CovarianceFunction& covFunc;
    
    unsigned int     sizeActiveSet;
    unsigned int     maxActiveSet;
    double  epsilonTolerance;
    bool    momentProjection;
    int     iterChanging;
    int     iterFixed;
    
    // Elements of computation
    mat KB;             // covariance between BV
    mat Q;              // inverse covariance between BV
    mat C;              // for calculating variance
    vec alpha;          // alphas for calculating mean
    
    double gammaTolerance;  // Threshold determining whether an observation is added to active set

    mat     ActiveSet;     // Active set
    uvec    idxActiveSet;  // Indexes of observations in active set

    mat P;              // projection coefficient matrix (full obs onto active set)
    vec meanEP;         // EP mean parameter (a)
    vec varEP;          // EP variance parameter(lambda)
    
    vec logZ;           // log-evidence

    // Augmented matrices - When doing a full update, we add and remove points to 
    // the active set. This operation being expensive (it involves memory reallocations),
    // it is more efficient to use temporary extended matrices for that purpose. These
    // matrices correspond to having an active set with an extra point. Points are added
    // to the extended matrices, and removed by swapping the relevant rows/columns with the
    // actual matrices (KB, Q, C, ...)
    mat KB_aug;
    mat Q_aug;
    mat C_aug;
    vec alpha_aug;
    mat ActiveSet_aug;
    uvec idxActiveSet_aug;
    mat P_aug;
    
    // Augmented stuff - version 2: only store the changes, not the full matrices
    // Modify the original ones instead.
    int idxActiveSet_new;         // Index of the latest active point added to the set
    rowvec ActiveSet_new;            // The latest (extra) active point added to the set
    vec P_new;                    // The column of P for the latest active point
    vec KB_new;                   // The covariance btw new active point and older ones
    double kb_new;                // The auto-covariance of the new active point 
    vec Q_new;                    // The inverse cov btw new active point and older ones
    double q_new;                 // The inverse auto-cov btw new active point and older ones
    double alpha_new;             // Alpha for the latest active point
    vec C_new;                    // C btw latest active point and older ones
    double c_new;                 // C btw latest active point and itself

    
    LikelihoodCalculation likelihoodType;
    
    
    
	// These methods provide the core algorithm
    void processObservationEP(const unsigned int iObs, const LikelihoodType &noiseModel, const bool fixActiveSet);
    void EP_removePreviousContribution(unsigned int iObs);
    void EP_updateIntermediateComputations(double &cavityMean, double &cavityVar, double &sigmaLoc,
                                           vec &k, double &gamma, vec &eHat, vec loc);
    void EP_updateEPParameters(unsigned int iObs, double q, double r, double cavityMean, double cavityVar,
                               double logEvidence);
    void EP_removeCollapsedPoints();
    
    // ALGO_V1: Implementation of the add/remove active point, version 1
    void addActivePoint(unsigned int iObs, double q, double r, vec k, double sigmaLoc, double gamma, vec eHat);
    void deleteActivePoint(unsigned int iObs);
    
    // ALGO_V2: Implementation of the add/remove active point, version 2 (augmented matrices)
    void addActivePointAugmented_v1(unsigned int iObs, double q, double r, vec k, double sigmaLoc, double gamma, vec eHat);
    void swapActivePoint_v1(unsigned int iObs);

    // ALGO_V3: Implementation of the add/remove active point, version 3 (no augmented matrices)
    void addActivePointAugmented_v2(unsigned int iObs, double q, double r, vec k, double sigmaLoc, double gamma, vec eHat);
    void swapActivePoint_v2(unsigned int iObs);
    
    void stabiliseCoefficients(double& q, double& r, double cavityMean, double cavityVar, double upperTolerance, double lowerTolerance);
	vec scoreActivePoints(ScoringMethod sm);

	// Parameter optimisation functions and their gradients
	double compEvidence() const;
	double compEvidenceApproximate() const;
	double compEvidenceUpperBound() const;
	vec gradientEvidence() const;
	vec gradientEvidenceApproximate() const;
	vec gradientEvidenceUpperBound() const;

	// Numerical tools
	mat computeCholesky(const mat& iM) const;
	mat computeInverseFromCholesky(const mat& C) const;
};

#endif /*PSGP_H_*/
