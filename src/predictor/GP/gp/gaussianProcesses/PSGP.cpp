#include "PSGP.h"

/**
 * Constructor
 * 
 * Parameters
 * 
 * X             Matrix of inputs (locations)
 * Y             Vector outputs (observations)
 * nActivePoints Maximum number of active points 
 * _iterChancing Number of sweeps through the data with replacement 
 *               of active points (default: 1)
 * _iterFixed    Number of sweeps through the data with fixed active 
 *               set (default :2)
 */
PSGP::PSGP(mat& X, vec& Y, CovarianceFunction& cf, int nActivePoints,
		int _iterChanging, int _iterFixed) :
		ForwardModel(X.n_cols, 1), Locations(X), Observations(Y), covFunc(cf) {

	// The active set size cannot exceed the number of observations
	maxActiveSet = min(nActivePoints, (int) Observations.n_rows);
	epsilonTolerance = 1e-6;
	gammaTolerance = 1e-3;

	iterChanging = _iterChanging;
	iterFixed = _iterFixed;

	nObs = Locations.n_rows;

	likelihoodType = Approximate;

	resetPosterior();

	// Which version of the implementation to use. Should be using V3
	// as it is the fastest, but I have left the other versions for
	// backward comparison/debugging, should they be needed.
	algoVersion = ALGO_V3;

	display=true;

}

/**
 * Destructor
 */
PSGP::~PSGP() {
}

/**
 * Specify the active set (by indices).
 * This resets the current active set and initialises the posterior for
 * the specified active set.
 */
void PSGP::setActiveSet(uvec activeIndexes, mat activeLocations) {
	

	resetPosterior();

	// Set active indexes and locations
	idxActiveSet = activeIndexes;
	ActiveSet = activeLocations;
	sizeActiveSet = activeIndexes.n_elem;

	// Disable replacement of active set in posterior computation
	iterChanging = 0;
}

/**
 * Compute posterior with a single likelihood model
 */
void PSGP::computePosterior(const LikelihoodType& noiseModel) {
	bool fixActiveSet = false;

	// Cycle several times through the data, first allowing the active
	// set to change (for iterChanging iterations) and then fixing it
	// (for iterFixed iterations)
	for (int cycle = 1; cycle <= (iterChanging + iterFixed); cycle++) {
		if (cycle > iterChanging)
			fixActiveSet = true;

		// Present observations in a random order
		uvec randObsIndex = randperm(nObs);

		for (unsigned int i = 0; i < nObs; i++) {
			if(display){
			Rprintf("\rProcessing observation: %d/%d", i + 1, nObs);
			}
			processObservationEP(randObsIndex(i), noiseModel, fixActiveSet);
		}
	}
}

/**
 * Compute posterior with a different likelihood model for each observation
 * modelIndex maps from the models in noiseModel to the observations. Note that
 * one model can be used for several observations (duplicated index in modelIndex).
 */
void PSGP::computePosterior(const arma::ivec& modelIndex,
		const std::vector<LikelihoodType *> noiseModel) {

	bool fixActiveSet = false;

	// Cycle several times through the data, first allowing the active
	// set to change (for iterChanging iterations) and then fixing it
	// (for iterFixed iterations)
	for (int cycle = 1; cycle <= (iterChanging + iterFixed); cycle++) {
		if (cycle > iterChanging)
			fixActiveSet = true;

		// Present observations in a random order
		uvec randObsIndex = randperm(nObs);

		for (unsigned int iObs = 0; iObs < nObs; iObs++) {
			unsigned int iModel = modelIndex( randObsIndex(iObs) );

			

			processObservationEP(randObsIndex(iObs), *noiseModel[iModel],
					fixActiveSet);
		}
	}
}

/**
 * This is the core method, implementing the sparse EP algorithm
 * 
 * This algorithm follows Appendix G in in Lehel Csato's PhD thesis 
 * "Gaussian Processes - Iterative Sparse Approximations", 2002,
 * NCRG, Aston University.
 * 
 * Further reference numbers correspond to sections/equations in the same
 * document. 
 */
void PSGP::processObservationEP(const unsigned int iObs,
		const LikelihoodType &noiseModel, const bool fixActiveSet) {
	double sigmaLoc; // Auto-covariance of location
	vec k = psgp_zeros(sizeActiveSet); // Covariance between location and active set

	double r, q; // Online coefficients
	double cavityMean, cavityVar; // Cavity mean and variance
	double gamma; // Mean of ??
	vec eHat; // Variance of ??

	// Retrieve location and observation at index iObs
	vec loc = Locations.row(iObs).t();
	double obs = Observations(iObs);

	// Remove previous contribution of observation iObs
	// Appendix G.(a)
	EP_removePreviousContribution(iObs);

	// Intermediate computations: covariances, cavity mean and variance
	// Appendix G.(c)
	EP_updateIntermediateComputations(cavityMean, cavityVar, sigmaLoc, k, gamma,
			eHat, loc);

	// Compute the updated q and r online coefficients for the specified
	// likelihood model, using the updated alpha and C.
	// Appendix G.(b), Sec. 2.4, Eq. 2.42, 3.28
	double logEvidence = noiseModel.updateCoefficients(q, r, obs, cavityMean,
			cavityVar);

	// stabiliseCoefficients(q, r, cavityMean, cavityVar, 1e3, 1e-6);

	// Update EP parameters
	EP_updateEPParameters(iObs, q, r, cavityMean, cavityVar, logEvidence);

	// Perform full or sparse update depending on geometry (gamma)
	// Appendix G.(e)
	if (gamma >= gammaTolerance * sigmaLoc && !fixActiveSet) {
		//----------------------------------------------
		// Full update
		//----------------------------------------------

		if (sizeActiveSet < maxActiveSet) {
			// Add observation to active set and update parameters
			// Rprintf(" Adding active point\n");
			addActivePoint(iObs, q, r, k, sigmaLoc, gamma, eHat);
		} else {
			switch (algoVersion) {

			case ALGO_V1: {
				// Add observation to active set
				addActivePoint(iObs, q, r, k, sigmaLoc, gamma, eHat);

				// Compute scores for each active point
				vec scores = scoreActivePoints(FullKL);

				// Remove active point with lowest score, and update alpha, C, Q and P
				unsigned int swapCandidate;
				scores.min(swapCandidate);
				deleteActivePoint(swapCandidate);
				break;
			}
			case ALGO_V2:
				// Swap observation with one existing active point if
				// the swap results in an improved active set
				addActivePointAugmented_v1(iObs, q, r, k, sigmaLoc, gamma,
						eHat);
				break;

			default: // This also covers the case ALGO_V3
				// Swap observation with one existing active point if
				// the swap results in an improved active set
				addActivePointAugmented_v2(iObs, q, r, k, sigmaLoc, gamma,
						eHat);
				break;
			}
		}

	} else {
		//----------------------------------------------
		// Sparse update
		//----------------------------------------------
		P.row(iObs) = eHat.t();

		vec s;
		if (sizeActiveSet > 0)
			s = C * k;
		s += eHat;

		// Update GP parameters - same as full update case, but with scaling
		// Appendix G.(f)
		double eta = 1.0 / (1.0 + gamma * r); // Scaling factor
		alpha += eta * s * q;
		// C     += r * eta * outer_product(s, s, false);
		C += r * eta * (s * s.t());
	}

	// Remove unneeded active points based on geometry
	EP_removeCollapsedPoints();
}

/**
 * Substract contribution of observation iObs from previous iterations 
 * 
 * Appendix G.(a), Eq. 4.19 and 4.20
 */
void PSGP::EP_removePreviousContribution(unsigned int iObs) {
	if (varEP(iObs) > LAMBDA_TOLERANCE) {
		vec p = P.row(iObs).t();
		vec Kp = KB * p;

		// Update alpha and C
		vec h = C * Kp + p;
		double nu = varEP(iObs) / (1.0 - varEP(iObs) * dot(Kp, h));
		alpha += h * nu * (dot(alpha, Kp) - meanEP(iObs));
		// C += nu * outer_product(h,h);
		C += nu * (h * h.t());
	}
}

/**
 * Compute cavity mean and variance for specified input/location x
 * 
 * Appendix G.(c), Eq. 3.3
 */
void PSGP::EP_updateIntermediateComputations(double &cavityMean,
		double &cavityVar, double &sigmaLoc, vec &k, double &gamma, vec &eHat,
		vec loc) {
	

	covFunc.computeSymmetric(sigmaLoc, loc); // Auto-variance of location

	if (sizeActiveSet == 0) {
		cavityVar = sigmaLoc;
		cavityMean = 0.0;
		eHat = psgp_zeros(0);
		gamma = sigmaLoc;
	} else {
		covFunc.computeSymmetric(sigmaLoc, loc); // Auto-variance of location
		covFunc.computeCovariance(k, ActiveSet, loc); // cov(location, active set)
		cavityVar = sigmaLoc + dot(k, C * k);
		cavityMean = dot(k, alpha);

		// This way of computing eHat is more robust - but much more
		// computationaly expensive, as we already store Q = inv(KB)
		// eHat = backslash(KB, k);
		eHat = Q * k;
		gamma = sigmaLoc - dot(k, eHat);
	}
}

/**
 * Update the EP parameters (Eq. 4.18)
 */
void PSGP::EP_updateEPParameters(unsigned int iObs, double q, double r,
		double cavityMean, double cavityVar, double logEvidence) {
	double ratio = q / r;
	logZ(iObs) = logEvidence
			+ (log(2.0 * arma::math::pi()) - log(abs(r)) - (q * ratio)) / 2.0;
	meanEP(iObs) = cavityMean - ratio;
	varEP(iObs) = -r / (1.0 + (r * cavityVar));
}

/**
 * Add a given location to the active set
 */
void PSGP::addActivePoint(unsigned int iObs, double q, double r, vec k, double sigmaLoc,
		double gamma, vec eHat) {
	vec s;
	if (sizeActiveSet > 0)
		s = C * k;

	// Increase size of active set
	sizeActiveSet++;

	// Append index of observation to indexes in active set
	idxActiveSet.resize(sizeActiveSet);
	idxActiveSet(sizeActiveSet - 1) = iObs;

	// Increase storage size for active set and store new observation
	ActiveSet = arma::join_cols(ActiveSet, Locations.row(iObs));

	// Increase size of C and alpha
	alpha.resize(sizeActiveSet);
	C.resize(sizeActiveSet, sizeActiveSet);

	// e is the unit vector for dimension sizeActiveSet
	vec e = psgp_zeros(nObs);
	e(iObs) = 1.0;

	// Update P matrix
	//    P.append_col(e);
	P = arma::join_rows(P, e);

	// Update KB matrix - add auto and cross covariances
	// KB.append_col(k);
	KB = arma::join_rows(KB, k);
	// KB.append_row( concat(k, sigmaLoc) );
	KB = arma::join_cols(KB,
			arma::join_rows(k.t(), sigmaLoc * arma::ones(1, 1)));

	// update Q = inv(KB) matrix
	eHat.resize(sizeActiveSet);
	eHat(sizeActiveSet - 1) = -1.0;
	Q.resize(sizeActiveSet, sizeActiveSet);
	// Q += outer_product(eHat, eHat) / gamma;
	Q += (eHat * eHat.t()) / gamma;

	// Update GP parameters
	// Appendix G.(f)
	s.resize(sizeActiveSet, true); // Increase size of s and append 1.0
	s(sizeActiveSet - 1) = 1.0;

	alpha += s * q;
	C += r * (s * s.t());
}

/**
 * Swap current observation with an existing active point if
 * this results in an improved active set.
 * 
 * What we do is build an extended active set (of size maxActiveSet+1) and then remove
 * the active point with the lowest score to get back to size maxActiveSet. Instead of
 * adding/removing, we use temporary augmented matrices and only update the original
 * matrices if a swap is necessary. This avoids expensive memory operations. 
 */
void PSGP::addActivePointAugmented_v1(unsigned int iObs, double q, double r, vec k,
		double sigmaLoc, double gamma, vec eHat) {
	

	// Initialise augmented matrices
	vec z = psgp_zeros(maxActiveSet + 1);

	idxActiveSet_aug = uvec(idxActiveSet.n_elem + 1);
	idxActiveSet_aug.rows(0, idxActiveSet.n_elem - 1) = idxActiveSet;
	idxActiveSet_aug.row(idxActiveSet.n_elem) = iObs;

	alpha_aug = vec(alpha.n_elem);
	alpha_aug.rows(0, alpha.n_elem - 1) = alpha;
	alpha_aug.row(alpha.n_elem) = 0.0;

	int n = sizeActiveSet - 1;
	ActiveSet_aug.submat(0, n, 0, n) = ActiveSet;
	P_aug.submat(0, n, 0, n) = P;
	Q_aug.submat(0, n, 0, n) = Q;
	C_aug.submat(0, n, 0, n) = C;
	KB_aug.submat(0, n, 0, n) = KB;

	Q_aug.row(maxActiveSet) = z.t();
	Q_aug.col(maxActiveSet) = z;
	C_aug.row(maxActiveSet) = z.t();
	C_aug.col(maxActiveSet) = z;

	// Add current observation to augmented active set
	ActiveSet_aug.row(maxActiveSet) = Locations.row(iObs);

	// e is the unit vector for dimension sizeActiveSet
	vec e = psgp_zeros(nObs);
	e(iObs) = 1.0;
	P_aug.col(maxActiveSet) = e;

	// Update KB matrix - add auto and cross covariances
	vec newk = vec(k.n_elem + 1);
	newk.rows(0, k.n_elem - 1) = k;
	newk.row(k.n_elem) = sigmaLoc;
	KB_aug.col(maxActiveSet) = newk;
	KB_aug.row(maxActiveSet) = newk.t();

	// update Q = inv(KB) matrix
	vec eHat_aug = vec(eHat.n_elem);
	eHat_aug.rows(0, eHat.n_elem - 1) = eHat;
	eHat_aug.row(eHat.n_elem) = -1.0;
	Q_aug += (eHat_aug * eHat_aug.t()) / gamma;

	// Update GP parameters
	// Appendix G.(f)
	vec s = arma::join_cols(C * k, arma::ones(1, 1));

	alpha_aug += s * q;
	C_aug += r * (s * s.t());

	// Determine which point in the extended active set we need to remove
	// to get back to the maximum size. If this point is not the point we
	// we just added, swap it with the current observation.

	// Compute scores for each active point
	vec scores = scoreActivePoints(FullKL);

	// Remove active point with lowest score, and update alpha, C, Q and P
	unsigned int swapCandidate;
	scores.min(swapCandidate);
	swapActivePoint_v1(swapCandidate);
}

/**
 * Swap current observation with an existing active point if
 * this results in an improved active set.
 * 
 * Typically, we:
 * - add a new observation to the active set (but store its effect 
 *   separately)
 * - remove one observation from the new, extended active set (to
 *   get back to the maximum size allowe)
 * - update the C and alpha to reflect the addition/removal of an 
 *   observation. However, we do it in such a way that there is minimal
 *   use of memory and no resizing of C and alpha.
 */
void PSGP::addActivePointAugmented_v2(unsigned int iObs, double q, double r, vec k,
		double sigmaLoc, double gamma, vec eHat) {
	


	ActiveSet_new = Locations.row(iObs);
	idxActiveSet_new = iObs;

	P_new = psgp_zeros(nObs);
	P_new(iObs) = 1.0;

	alpha_new = q;

	vec s = C * k;
	C_new = r * s;
	c_new = r;

	Q_new = -eHat / gamma;
	q_new = 1.0 / gamma;

	KB_new = k;
	kb_new = sigmaLoc;

	// Update parameters for current active set
	alpha += q * s;
	C += r * (s * s.t());
	Q += (eHat * eHat.t()) / gamma;

	// Determine which point in the extended active set we need to remove
	// to get back to the maximum size. If this point is not the point we
	// we just added, swap it with the current observation.

	// Compute scores for each active point
	vec scores = scoreActivePoints(FullKL);

	// Remove active point with lowest score, and update alpha, C, Q and P
	// int swapCandidate = min_index(scores);
	unsigned int swapCandidate;
	scores.min(swapCandidate);
	swapActivePoint_v2(swapCandidate);
}

/**
 * Delete an active point from the active set
 *
 * @params:
 *  iObs    The index of the active point to be deleted
 */
void PSGP::deleteActivePoint(unsigned int iObs) {
	// Elements for iObs (correspond to the * superscripts in Csato)
	double alpha_i = alpha(iObs);
	double c_i = C(iObs, iObs);
	double q_i = Q(iObs, iObs);
	vec P_i = P.col(iObs);

	// Covariance between iObs and other active points
	vec C_i = C.row(iObs).t();
	vec Q_i = Q.row(iObs).t();
	C_i.shed_col(iObs); // Delete cov(iObs, iObs), we only
	Q_i.shed_col(iObs); // want the cross terms

	// Updated elements without iObs (correspond to the "r" superscripts in Csato)
	alpha.shed_col(iObs);
	C.shed_col(iObs);
	C.shed_row(iObs);
	Q.shed_col(iObs);
	Q.shed_row(iObs);
	P.shed_col(iObs);

	// Update new (reduced) elements
	alpha -= (alpha_i / (c_i + q_i)) * (Q_i + C_i); // Eq. 3.19
	mat QQq = (Q_i * Q_i.t()) / q_i;
	C += QQq - (Q_i + C_i) * (Q_i + C_i).t() / (q_i + c_i);
	Q -= QQq;
	P -= (P_i * Q_i.t()) / q_i;

	// Update Gram matrix
	KB.shed_row(iObs);
	KB.shed_col(iObs);

	// Update active set
	ActiveSet.shed_row(iObs);
	idxActiveSet.shed_row(iObs);
	sizeActiveSet--;
}

/**
 * Swap specified active point (iObs) with new active point 
 * (last index in augmented matrices)
 */
void PSGP::swapActivePoint_v1(unsigned int iDel) {
	// Swap iObs with last active point in augmented matrices
	unsigned int iObs = maxActiveSet;

	if (iDel != maxActiveSet) {
		C_aug.swap_cols(iObs, iDel);
		C_aug.swap_rows(iObs, iDel);
		Q_aug.swap_cols(iObs, iDel);
		Q_aug.swap_rows(iObs, iDel);
		KB_aug.swap_cols(iObs, iDel);
		KB_aug.swap_rows(iObs, iDel);
		P_aug.swap_cols(iObs, iDel);
		double alpha_last = alpha_aug(iObs);
		alpha_aug(iObs) = alpha_aug(iDel);
		alpha_aug(iDel) = alpha_last;
		P.col(iDel) = P_aug.col(iDel);

		// Update Gram matrix
		vec k_add = KB_aug.col(iDel);
		k_add.shed_row(iObs);

		KB.col(iDel) = k_add;
		KB.row(iDel) = k_add.t();

		// Update active set
		ActiveSet.row(iDel) = ActiveSet_aug.row(maxActiveSet);
		idxActiveSet(iDel) = idxActiveSet_aug(maxActiveSet);
	}

	alpha = alpha_aug(0, iObs - 1);
	C = C_aug.submat(0, iObs - 1, 0, iObs - 1);
	Q = Q_aug.submat(0, iObs - 1, 0, iObs - 1);

	// Elements for point to be deleted (correspond to the * superscripts in Csato)
	double alpha_i = alpha_aug(iObs);

	double c_i = C_aug(iObs, iObs);
	double q_i = Q_aug(iObs, iObs);
	vec P_i = P_aug.col(iObs);

	// Covariance between element to be removed and other active points
	vec C_i = C_aug.row(iObs).t();
	vec Q_i = Q_aug.row(iObs).t();
	C_i.shed_row(iObs); // Delete cov(iObs, iObs), we only
	Q_i.shed_row(iObs); // want the cross terms

	// Update new (reduced) elements
	alpha -= (alpha_i / (c_i + q_i)) * (Q_i + C_i); // Eq. 3.19
	mat QQq = (Q_i * Q_i.t()) / q_i;
	C += QQq - (Q_i + C_i) * (Q_i + C_i).t() / (q_i + c_i);
	Q -= QQq;
	P -= (P_i * Q_i.t()) / q_i;
}

/**
 * Replaces given active point (iDel) with new active point,
 * for which data has been stored previously (in addActivePointAugmented)
 * in the vectors and matrices *_new.
 * 
 * We proceed by swapping the point to be deleted with the new active point,
 * and then perform the update as in Csato. The advantage of this method is
 * it does not requires (expensive) resizing of matrices.
 */
void PSGP::swapActivePoint_v2(unsigned int iDel) {
	double alpha_i, c_i, q_i;
	arma::colvec P_i, C_i, Q_i;

	

	if (iDel == maxActiveSet) {
		alpha_i = alpha_new;
		q_i = q_new;
		c_i = c_new;

		P_i = P_new;
		Q_i = Q_new;
		C_i = C_new;
	} else {
		// Elements for point to be deleted (correspond to the * superscripts in Csato)
		alpha_i = alpha(iDel);
		c_i = C(iDel, iDel);
		q_i = Q(iDel, iDel);
		P_i = P.col(iDel);

		// Covariance between element to be removed and other active points
		C_i = C.row(iDel).t();
		Q_i = Q.row(iDel).t();

		C_i(iDel) = C_new(iDel); // Replace cov(iDel, iDel) with
		Q_i(iDel) = Q_new(iDel); // cov(iDel, new point)

		C_new(iDel) = c_new; // cov(iDel, new point) is now
		Q_new(iDel) = q_new; // cov(new point, new point)

		// Swap point to be removed with new point
		alpha(iDel) = alpha_new;

		C.col(iDel) = C_new;
		C.row(iDel) = C_new.t();

		Q.col(iDel) = Q_new;
		Q.row(iDel) = Q_new.t();

		P.col(iDel) = P_new;

		// Update Gram matrix
		KB_new(iDel) = kb_new;
		KB.col(iDel) = KB_new;
		KB.row(iDel) = KB_new.t();

		// Update active set
		ActiveSet.row(iDel) = ActiveSet_new;
		idxActiveSet(iDel) = idxActiveSet_new;
	}

	// Update new (reduced) elements
	alpha -= (alpha_i / (c_i + q_i)) * (Q_i + C_i); // Eq. 3.19
	mat QQq = (Q_i * Q_i.t()) / q_i;
	C += QQq - (Q_i + C_i) * (Q_i + C_i).t() / (q_i + c_i);
	Q -= QQq;
	P -= P_i * Q_i.t() / q_i;
}

/**
 * Remove active points that might have become unnecessary (based on
 * geometry criterion)  
 */
void PSGP::EP_removeCollapsedPoints() {
	while (sizeActiveSet > 0) {
		vec scores = scoreActivePoints(Geometric);
		unsigned int removalCandidate;
		scores.min(removalCandidate);

		if (scores(removalCandidate) >= (gammaTolerance / 1000.0)) {
			break;
		}
		deleteActivePoint(removalCandidate);
	}
}

/**
 * Score active points according to scoring method
 */
vec PSGP::scoreActivePoints(ScoringMethod sm) {
	vec diagC, diagS, term1, term2, term3;
	vec diagInvGram, a;

	// Extract relevant part from either augmented matrices or original ones
	switch (algoVersion) {
	case ALGO_V1:
		a = alpha;
		diagC = arma::diagvec(C);
		diagInvGram = arma::diagvec(Q);
		break;

	case ALGO_V2:
		diagInvGram = arma::diagvec(Q_aug);
		a = alpha_aug;
		diagC = arma::diagvec(C_aug);
		break;

	default: // This also covers the case ALGO_V3
		a = join_cols(alpha, alpha_new * arma::ones(1, 1));
		diagC = join_cols(arma::diagvec(C), c_new * arma::ones(1, 1));
		diagInvGram = join_cols(arma::diagvec(Q), q_new * arma::ones(1, 1));
		break;
	}

	switch (sm) {
	case Geometric:
		return (1.0 / diagInvGram);

	case MeanComponent:
		return (a % a) / (diagC + diagInvGram);

	case FullKL: // Lehel: Eq. 3.23
	{
		switch (algoVersion) {
		case ALGO_V1:
			// diagS = diag((P.transpose() * diag(varEP)) * P);
			diagS = psgp_zeros(P.n_cols);
			for (unsigned int i = 0; i < P.n_cols; i++) {
				diagS(i) = arma::accu(varEP % (P.col(i) % P.col(i)));
			}
			break;

		case ALGO_V2:
			diagS = psgp_zeros(P_aug.n_cols);
			for (unsigned int i = 0; i < P_aug.n_cols; i++) {
				diagS(i) = arma::accu(varEP % (P_aug.col(i) % P_aug.col(i)));
			}
			break;

		default: // This also covers the case ALGO_V3
			diagS = psgp_zeros(P.n_cols + 1);
			for (unsigned int i = 0; i < P.n_cols; i++) {
				diagS(i) = arma::accu(varEP % (P.col(i) % P.col(i)));
			}
			diagS(P.n_cols) = arma::accu(varEP % (P_new % P_new));
		}

		term1 = (a % a) / (diagC + diagInvGram);
		term2 = diagS / diagInvGram;
		term3 = arma::log(1.0 + (diagC / diagInvGram));
		return (term1 + term2 + term3);
	}

	default:
		Rprintf("Unknown scoring method\n");
		break;
	}

	return zeros(diagInvGram.n_elem);
}

/**
 * Check update coefficients to ensure stability
 */
void PSGP::stabiliseCoefficients(double& q, double& r, double cavityMean,
		double cavityVar, double upperTolerance, double lowerTolerance) {
	double sqrtPt = sqrt(cavityVar);
	double tu = -sqrtPt * r * sqrtPt;
	bool mod = false;
	if (tu > upperTolerance) {
		tu = upperTolerance;
		mod = true;
	}

	if (tu < lowerTolerance) {
		tu = lowerTolerance;
		mod = true;
	}

	if (mod) {
		r = -(tu / sqrtPt) / tu;
		r = r + arma::math::eps();
		r = r + r;
	}
}

/**
 * Make predictions
 */
void PSGP::makePredictions(vec& Mean, vec& Variance, const mat& Xpred,
		CovarianceFunction& cf) const {
	
	

	// Predictive mean
	mat ktest(Xpred.n_rows, sizeActiveSet);
	cf.computeCovariance(ktest, Xpred, ActiveSet);
	Mean = ktest * alpha;

	// Predictive variance
	vec kstar(Xpred.n_rows);
	cf.computeDiagonal(kstar, Xpred);
	Variance = kstar + arma::sum((ktest * C) % ktest, 1);




}

/**
 * Same as above, but using the current (stored) covariance function to make
 * the predictions.
 **/
void PSGP::makePredictions(vec& Mean, vec& Variance, const mat& Xpred) const {
	makePredictions(Mean, Variance, Xpred, covFunc);
}

/**
 * Simulate from PSGP
 */
vec PSGP::simulate(const mat& Xpred, bool approx) const {
	mat cov, vCov, kxbv(Xpred.n_rows, sizeActiveSet);
	vec dCov, samp;

	covFunc.computeCovariance(kxbv, Xpred, ActiveSet);

	if (approx) {
		cov = Q + C;
		vCov.resize(sizeActiveSet, sizeActiveSet); // Eigen vectors
		dCov.resize(sizeActiveSet); // Eigen values
		arma::eig_sym(dCov, vCov, cov);
		vCov = kxbv * vCov;
		samp = arma::randn(sizeActiveSet);
	} else {
		mat kxx(Xpred.n_rows, Xpred.n_rows);
		covFunc.computeSymmetric(kxx, Xpred);
		cov = kxx + ((kxbv * C) * kxbv.t());
		eig_sym(dCov, vCov, cov);
		samp = arma::randn(Xpred.n_rows);
	}

	dCov = sqrt(abs(dCov));

	vec a1 = kxbv * alpha;
	mat a2 = vCov * arma::diagmat(dCov);

	return (a1 + (a2 * samp));
}

/**
 * Get covariance function parameters
 */
vec PSGP::getParametersVector() const {
	return covFunc.getParameters();
}

/**
 * Set covariance function parameters
 */
void PSGP::setParametersVector(const vec p) {
	covFunc.setParameters(p);
}

/**
 * Recompute posterior parameters
 */
void PSGP::recomputePosterior() {

	mat KBold = KB;
	mat Kplus(Observations.n_elem, sizeActiveSet);
	covFunc.computeSymmetric(KB, ActiveSet);
	covFunc.computeCovariance(Kplus, Locations, ActiveSet);

	// RB: P should be transpose(inv(KB)*Kplus), not inv(KB)*Kplus (size is wrong otherwise)
	mat Ptrans(P.n_cols, P.n_rows);
	Ptrans = solve(KB, Kplus.t());
	P = Ptrans.t();

	mat varEPdiag = arma::diagmat(varEP);
	mat projLam = P.t() * varEPdiag;
	mat UU = projLam * P;
	mat CC = UU * KB + arma::eye(sizeActiveSet, sizeActiveSet);
	alpha = solve(CC, projLam * meanEP);
	C = -solve(CC, UU);
	Q = computeInverseFromCholesky(KB);
}

/**
 * Reset posterior representation
 */
void PSGP::resetPosterior() {
	KB.resize(0, 0);
	Q.resize(0, 0);
	C.resize(0, 0);
	alpha.resize(0);
	ActiveSet.resize(0, getInputDimensions());
	idxActiveSet.resize(0);
	sizeActiveSet = 0;
	P = psgp_zeros(Observations.n_elem, 0);

	KB_aug = psgp_zeros(maxActiveSet + 1, maxActiveSet + 1);
	Q_aug = psgp_zeros(maxActiveSet + 1, maxActiveSet + 1);
	C_aug = psgp_zeros(maxActiveSet + 1, maxActiveSet + 1);
	alpha_aug = psgp_zeros(maxActiveSet + 1);
	ActiveSet_aug = psgp_zeros(maxActiveSet + 1, getInputDimensions());
	idxActiveSet_aug.zeros(maxActiveSet + 1);
	P_aug = psgp_zeros(Observations.n_elem, maxActiveSet + 1);

	alpha_new = 0.0;
	kb_new = 0.0;
	c_new = 0.0;
	q_new = 0.0;
	idxActiveSet_new = -1;
	ActiveSet_new = rowvec().zeros(getInputDimensions());//zeros(getInputDimensions());
	C_new = psgp_zeros(maxActiveSet - 1);
	KB_new = psgp_zeros(maxActiveSet - 1);
	Q_new = psgp_zeros(maxActiveSet - 1);
	P_new = psgp_zeros(Observations.n_elem);

	varEP = psgp_zeros(Observations.n_elem);
	meanEP = psgp_zeros(Observations.n_elem);
	logZ = psgp_zeros(Observations.n_elem);

}

/**
 * Objective function
 * 
 * RB: Can we possibly replace the switch() with an OO version? I.e.
 * have an Evidence class which is extended by EvidenceFull, 
 * EvidenceApproximate and EvidenceUpperBound? This would also solve the issue
 * of an unknown evidence model.
 */
double PSGP::objective() const {
	double evidence;

	switch (likelihoodType) {
	case FullEvid:
		evidence = compEvidence();
		break;

	case Approximate:
		evidence = compEvidenceApproximate();
		break;

	case UpperBound:
		evidence = compEvidenceUpperBound();
		break;

	default:
		// RB: This really ought to throw an exception
		Rprintf("Error in PSGP::objective: Unknown likelihood type.");
		return 0.0;
	}
	return evidence;
}

/**
 * Gradient of the objective function
 *
 * RB: Can we possibly replace the switch() with an OO version? I.e.
 * have an Evidence class which is extended by EvidenceFull, 
 * EvidenceApproximate and EvidenceUpperBound? Would also solve the issue
 * of an unknown evidence model.
 
 */
vec PSGP::gradient() const {
	vec g;

	switch (likelihoodType) {
	case FullEvid:
		g = gradientEvidence();
		break;

	case Approximate:
		g = gradientEvidenceApproximate();
		break;

	case UpperBound:
		g = gradientEvidenceUpperBound();
		break;

	default:
		// RB: This really ought to throw an exception
		g.resize(covFunc.getNumberParameters());
		break;
	}

	return g;
}

/**
 * Full evidence for current covariance function
 */
double PSGP::compEvidence() const {

	arma::cx_vec eval;
	arma::cx_mat evec;

	mat KB_new(sizeActiveSet, sizeActiveSet);

	covFunc.computeSymmetric(KB_new, ActiveSet);

	double evid = arma::accu(arma::log(varEP));

	evid -= arma::accu(arma::square(meanEP) % varEP);
	evid += 2.0 * sum(logZ);
	evid -= varEP.n_elem * log(2.0 * arma::math::pi());
	mat Klp = P.t() * arma::diagmat(varEP);
	mat Ksm = (Klp * P) * KB_new + arma::eye(sizeActiveSet, sizeActiveSet);
	vec Kall = Klp * meanEP;
	mat Kinv = solve(Ksm.t(), KB_new.t());

	evid += arma::dot(Kall, Kinv.t() * Kall);

	if (arma::eig_gen(eval, evec, Ksm)) {
		evid -= arma::accu(log(arma::real(eval)));
	} else {
		Rprintf("PSG:compEvidence: Error computing evidence\n");
	}

	return -evid / 2.0;
}

/**
 * Approximate evidence for current covariance function
 */
double PSGP::compEvidenceApproximate() const {
	mat cholSigma(sizeActiveSet, sizeActiveSet);
	mat Sigma(sizeActiveSet, sizeActiveSet);

	covFunc.computeSymmetric(Sigma, ActiveSet);
	mat invSigma = computeInverseFromCholesky(Sigma);

	vec obsActiveSet = Observations.elem(idxActiveSet);

	vec alpha = invSigma * obsActiveSet;

	double like1 = arma::accu(arma::log(arma::diagvec(arma::chol(Sigma))));
	double like2 = 0.5 * dot(obsActiveSet, alpha);

	return like1 + like2 + 0.5 * sizeActiveSet * log(2 * arma::math::pi());
}

/**
 * Upper bound on the evidence for current covariance function
 */
double PSGP::compEvidenceUpperBound() const {
	mat KB_new(sizeActiveSet, sizeActiveSet);
	covFunc.computeSymmetric(KB_new, ActiveSet);

	mat U(KB_new.n_rows, KB_new.n_cols);
	try {
		U = arma::chol(KB_new);
	} catch (std::runtime_error &e) {
		Rprintf("** Error: Cholesky decomposition of KB_new failed.");
		Rprintf("Current covariance function parameters:\n");
		covFunc.displayCovarianceParameters();
	}

	double like1 = 2.0 * arma::accu(arma::log(arma::diagvec(U)));
	// double like2 = trace((eye(sizeActiveSet) +
	//         (KB * (C + outer_product(alpha, alpha)))) * backslash(KB_new, KB));
	double like2 = arma::trace(
			(arma::eye(sizeActiveSet, sizeActiveSet)
					+ KB * (C + alpha * alpha.t()))
					* solve(U, solve(U.t(), KB)));

	return 0.5 * (like1 + like2 + sizeActiveSet * log(2 * arma::math::pi()));
}

/**
 * Gradient of full evidence
 */
vec PSGP::gradientEvidence() const {
	vec grads = psgp_zeros(covFunc.getNumberParameters());
	return grads;

}

/**
 * Gradient of approximate evidence
 */
vec PSGP::gradientEvidenceApproximate() const {
	vec grads(covFunc.getNumberParameters());

	mat cholSigma(sizeActiveSet, sizeActiveSet);
	mat Sigma(sizeActiveSet, sizeActiveSet);

	covFunc.computeSymmetric(Sigma, ActiveSet);
	cholSigma = computeCholesky(Sigma);
	mat invSigma = computeInverseFromCholesky(Sigma);
	// chol(Sigma, cholSigma);
	// mat invSigma = backslash( cholSigma, eye(sizeActiveSet) );
	// invSigma *= invSigma.transpose();

	vec obsActiveSet = Observations.elem(idxActiveSet);
	vec alpha = invSigma * obsActiveSet;

	mat W = (invSigma - alpha * alpha.t());

	mat partialDeriv(sizeActiveSet, sizeActiveSet);

	for (unsigned int i = 0; i < covFunc.getNumberParameters(); i++) {
		covFunc.getParameterPartialDerivative(partialDeriv, i, ActiveSet);
		grads(i) = arma::accu(W % partialDeriv) / 2.0;
	}
	return grads;
}

/**
 * Gradient of upper bound on evidence
 */
vec PSGP::gradientEvidenceUpperBound() const {

	vec grads(covFunc.getNumberParameters());

	mat W = arma::eye(sizeActiveSet, sizeActiveSet);
	mat KB_new(sizeActiveSet, sizeActiveSet);
	covFunc.computeSymmetric(KB_new, ActiveSet);

	// RB: This gives the correct gradient for the length scale
	mat partialDeriv(sizeActiveSet, sizeActiveSet);
	mat U = solve(KB_new, KB);

	W += KB * (C + alpha * alpha.t());

	for (unsigned int i = 0; i < covFunc.getNumberParameters(); i++) {
		covFunc.getParameterPartialDerivative(partialDeriv, i, ActiveSet);
		mat V1 = solve(KB_new, partialDeriv);
		mat V2 = W * solve(KB_new, partialDeriv * U);

		grads(i) = arma::trace(V1 - V2);
	}

	return 0.5 * grads;
}

/**
 * Set the likelihood type
 */
void PSGP::setLikelihoodType(LikelihoodCalculation lc) {
	likelihoodType = lc;
}

/**
 * Display current parameters
 */
void PSGP::displayModelParameters() const {
	Rprintf("Summary Sequential Gaussian Process\n");
	Rprintf("  Kernel Matrix size         : %dx%d\n", KB.n_rows, KB.n_cols);
	Rprintf("  Inverse Kernel Matrix size : %dx%d\n", Q.n_rows, Q.n_cols);
	Rprintf("  alpha size                 : %d\n", alpha.n_rows);
	Rprintf("  C size                     : %dx%d\n", C.n_rows, C.n_cols);
	Rprintf("  Projection matrix size     : %dx%d\n", P.n_rows, P.n_cols);
	Rprintf("  Lambda                     : %d\n", varEP.n_rows);
	Rprintf("  projection alpha           : %d\n", meanEP.n_rows);
	Rprintf("  log evidence vector        : %d\n", logZ.n_rows);
	Rprintf("  ----------------------------\n");
	Rprintf("  Predicion locations        : %dx%d\n", Locations.n_rows, Locations.n_cols);
	Rprintf("  Observations               : %d\n", Observations.n_rows);
	Rprintf("  Active set size            : %d (max %d)\n", ActiveSet.n_rows, maxActiveSet);
	Rprintf("  Epsilon tolerance          : %f\n", epsilonTolerance);
	Rprintf("  Iterations Changing/Fixed  : %d/%d\n", iterChanging, iterFixed);
}

/**
 * Compute Cholesky decomposition of matrix
 * 
 * RB: This belongs in a different class/library, really...
 * TODO: Move to ITPPExt
 */
mat PSGP::computeCholesky(const mat& iM) const {
	mat M = iM;
	

	const double ampl = 1.0e-10;
	const int maxAttempts = 10;

	mat cholFactor(M.n_rows, M.n_cols);

	int l = 0;

	while (l < maxAttempts) {
		try {
			return arma::chol(M);
		} catch (std::runtime_error &e) {
			Rprintf("Convergence error in computeCholesky.");
			Rprintf("Adding white noise to the diagonal to improve conditioning");
			double noiseFactor = abs(ampl * (trace(M) / double(M.n_rows)));
			M += noiseFactor * arma::eye(M.n_rows, M.n_rows);
			l++;
		}
	}

	Rprintf("Cholesky decomposition failed after %d attempts to improve conditioning.", l);
	Rprintf("PSGP results will be wrong.");
	return M;  // Return something so R doesn't crash...
}

/**
 * Compute inverse of a square matrix using Cholesky decomposition
 * 
 * TODO: Move to ITPPExt
 */
mat PSGP::computeInverseFromCholesky(const mat& C) const {
	mat cholFactor = computeCholesky(C);
	mat invChol = solve(cholFactor,
			arma::eye(cholFactor.n_rows, cholFactor.n_rows));
	return invChol * invChol.t();
}

