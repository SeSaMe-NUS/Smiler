/*
 * ParabalaFunc.h
 *
 *  Created on: Aug 13, 2014
 *      Author: zhoujingbo
 */

#ifndef PARABALAFUNC_H_
#define PARABALAFUNC_H_

#include "ForwardModel.h"
#include "../optimisation/Optimisable.h"
#include "../covarianceFunctions/CovarianceFunction.h"
#include "../covarianceFunctions/Transform.h"
#include "../psgp_common.h"


/**
 * todo:
 * y = x^2-2x+1
 */
class ParabalaFunc:public Optimisable {
public:
	ParabalaFunc();
	virtual ~ParabalaFunc();

	 double objective() const;
	 vec gradient() const ;

	 vec getParametersVector() const;
	 void setParametersVector(const vec p) ;


		 vec para;
};

#endif /* PARABALAFUNC_H_ */
