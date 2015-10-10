SMiLer: A Semi-Lazy Time Series Prediction System for Sensors
===
### 1. Introduction

SMiLer is a SeMiLazy time series prediction system for sensors. The overall framework of SMiLer comprising of two main steps: search step and prediction step. More details about it can be found in the paper: 

Jingbo Zhou, Anthony K. H. Tung; "SMiLer: A Semi-Lazy Time Series Prediction System for Sensors"; Proc. of 2015 ACM Int. Conf. on Management of Data (SIGMOD 2015)

### 2. Quick Run
You can see a running example in the folder "src/demo". The demo also give a brief introudction about the API and parameters.



### 3. Usage

This section shows important functional APIs of SMiLer.

#### 3.1 Data loading function

```cpp

	void conf_DataEngine(vector<vector<float> >& in_bladeData_vec, 
				int sc_band, int winDim,int maxOffset)
```

This function is in src/smilerManager/TSManagerOnGpuIdx.h.  It is the main function to load the sensor time series data into the GPU. The parameters of conf_DataEngine() are as follows:

##### 3.1.1 Parameters of conf_DataEngine():

	 'in_bladeData_vec' -- the time series of sensors, one blade is for one sensor
	 'sc_band' -- the warping width for LB_keogh of DTW
	 'winDim' -- the window length
	 'maxOffset' -- do not load maxOffset data into (or say left them out of) GPU to avoid exceeding the bounding when query the reference time series


#### 3.2 Continuous query function

```cpp

	void TSPred_continuous_onGPUIdx(
			vector<vector<float> >& groupQuery_vec, 
			int contPrdStep, 
			vector<int>& groupQuery_blade_map,
			int y_offset, int pred_selector, 
			bool weightedTrn, bool selfCorr)
```

This function is in src/smilerManager/TSManagerOnGpuIdx.h.  It is the main function to start the continuous query on SMiLer. By this function, SMiLer makes continuous prediction for each vector in groupQuery_vec. The parameters of TSPred_continuous_onGPUIdx() are as follows:

##### 3.2.1 Parameters TSPred_continuous_onGPUIdx():

	 'groupQuery_vec' -- a set of query vector for prediction, it can be multiple queries for mulitple sensors,
		the relations between sensors and queries are defined by a vector groupQuery_blade_map as show below
	 'contPrdStep' -- the step for making continuous prediction
	 'groupQuery_blade_map' -- map the query to corresponding data blade (one data blade is just the time series of one sensor)
	 'y_offset' -- the h-step ahead prediction for time series prediction.
	                       for one step ahead prediction,y_offset==0
	                       for multiple steps ahead prediction, y_offset = mul_step - 1
	 'pred_selector' -- to select the prediction model
	 			pred_selector = 1, smiler-AR
	 			pred_selector =0, smiler-GP (refer to section 5.2 of the paper SMiLer)
	 'weightedTrn' -- whether to do weighted training process, i.e. give different weight of the kNN results based DTW distance
	 'selfCorr' -- whether to use auto-tuning mechanism with self-adaptive prediction to determine the weight of K and L




### 4. Dependencies
The current code depends on the following external libraries:

Armadillo (http://arma.sourceforge.net/), is a high quality C++ linear algebra library.

OpenBLAS (optional, http://www.openblas.net/), is an optimized BLAS library. OpenBLAS is not mandatory, but it helps Armadillo achieve better performance.
(You also can use BLAS, ATLAS and LAPACK together to replace the OpenBLAS.)


***Note: The project is created by Nsight IDE.***


