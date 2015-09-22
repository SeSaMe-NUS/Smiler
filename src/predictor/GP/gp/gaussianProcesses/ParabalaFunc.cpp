/*
 * ParabalaFunc.cpp
 *
 *  Created on: Aug 13, 2014
 *      Author: zhoujingbo
 */

#include "ParabalaFunc.h"

ParabalaFunc::ParabalaFunc() {
	// TODO Auto-generated constructor stub
    para= vec().zeros(1);

    para(0)=10000;
}

ParabalaFunc::~ParabalaFunc() {
	// TODO Auto-generated destructor stub
}
/**
 * todo:
 * y = x^2-2x+1
 */
 double ParabalaFunc::objective() const{

	 return ((para(0)*para(0)-2*para(0)+1));

}
 vec ParabalaFunc::gradient() const {


	 vec o2 = 2*para -2;


	 return  o2;

	}

 vec ParabalaFunc::getParametersVector() const{

	 return para;

		}
 void ParabalaFunc::setParametersVector(const vec p) {
	 this->para=p;

		}
