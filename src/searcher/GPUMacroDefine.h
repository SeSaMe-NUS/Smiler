/*
 * GPUMacroDefine.h
 *
 *  Created on: Jun 20, 2014
 *      Author: zhoujingbo
 */

#ifndef GPUMACRODEFINE_H_
#define GPUMACRODEFINE_H_

#define THREAD_PER_BLK 256  // must be greater or equal to MAX_DIM and not exceed 1024

#define MAX_PRE_MUL_STEP 100 //	left MAX_PRE_MAX_STEP out of GPU to avoid exceeding the bounding when query the reference time series



#endif /* GPUMACRODEFINE_H_ */
