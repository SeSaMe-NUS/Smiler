#ifndef BUCKET_TOPK_INC_
#define BUCKET_TOPK_INC_

#include <iostream>

using namespace std;
#include <cuda_profiler_api.h>

static __global__ void default_start_index(int* start_index, int* end_index)
{
	int bid = blockIdx.x;
	start_index[bid] = (bid==0) ? 0 : end_index[bid-1];
}





template<typename Data, typename GetValue>
__host__ void tailored_bucket_topk(device_vector<Data> *d_data, GetValue sel, device_vector<float> *d_min, device_vector<float> *d_max, int k, device_vector<int> *d_start_index, device_vector<int> *d_end_index, int number_of_parts, device_vector<int> *d_result)
{
//	//for timing
//	cudaProfilerStart();
//	cudaEvent_t cuda_start;
//	cudaEventCreate(&cuda_start);
//	cudaEvent_t cuda_stop;
//	cudaEventCreate(&cuda_stop);
//	float timeRec[12];
//	for(int ti=0;ti<12;ti++){timeRec[ti]=0;}
//	float cuda_elapsedTime=0;
//	//for timing

//	cudaEventRecord(cuda_start, 0);//for timing
	//========================================================================================================================
	//device_vector<int> d_result_end_index(number_of_parts);
	//device_vector<int> d_result_last_index(number_of_parts);
	//inclusive_scan((*d_k).begin(), (*d_k).end(), d_result_end_index.begin());
	//exclusive_scan((*d_k).begin(), (*d_k).end(), d_result_last_index.begin());

	host_vector<int> h_result_last_index(number_of_parts);
	for(int i=0;i<number_of_parts;i++){
		h_result_last_index[i]=i*k;
	}

	device_vector<int> d_result_last_index = h_result_last_index;
	//device_vector<int> d_k(number_of_parts,k);

	const int size = (*d_end_index)[number_of_parts-1] - (*d_start_index)[0];
	//cout<<"bucket_topk d_bucket size:"<<size<<" mem:"<<size*sizeof(Bucket)<<" d_data size:"<<(*d_data).size()<<" mem:"<<(*d_data).size()*sizeof((*d_data)[0])<<" total mem:"<< size*sizeof(Bucket) +(*d_data).size()*sizeof((*d_data)[0]) <<endl;

	const int result_size = k*number_of_parts;//d_result_end_index[number_of_parts-1];
	int number_of_buckets = BUCKER_NUM_PER_PART < size/2 ? BUCKER_NUM_PER_PART : size/2+1;

	(*d_result).resize(result_size,-1);
	//thrust::fill((*d_result).begin(), (*d_result).end(), -1);
	//========================================================================================================================
//	 cudaEventRecord(cuda_stop, 0);//for timing
//	 cudaEventSynchronize(cuda_stop);
//	 cuda_elapsedTime=0;
//	 cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	 timeRec[0]=cuda_elapsedTime;

	//First Round
	//========================================================================================================================
//	 cudaEventRecord(cuda_start, 0);//for timing
	//1. Assign bucket index
	device_vector<Bucket> d_bucket(size);
	device_vector<int> d_bucket_size(number_of_parts*number_of_buckets);
	assign_bucket<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(*d_data), sel, rpc(d_bucket), rpc(*d_start_index), rpc(*d_end_index), rpc(d_bucket_size), rpc(*d_min), rpc(*d_max), number_of_buckets);
//	//for timing
//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[1]=cuda_elapsedTime;
//	//end for timing

//	 cudaEventRecord(cuda_start, 0);//for timing
	//2. Count last index and miss
	device_vector<int> d_last_bucket_index(number_of_parts);
	device_vector<int> d_miss(number_of_parts);
	device_vector<int> d_end_index_of_each_part(number_of_parts);
	tailored_count_sum(k, &d_bucket_size, number_of_parts, number_of_buckets, &d_last_bucket_index, &d_miss, &d_end_index_of_each_part);
//	//for timing
//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[2]=cuda_elapsedTime;
//	//end for timing

//	 cudaEventRecord(cuda_start, 0);//for timing
	//3. Tid partition
	device_vector<int> d_thread_start_index(number_of_parts*THREADS_PER_BLOCK);
	device_vector<int> d_thread_end_index(number_of_parts*THREADS_PER_BLOCK);
	tailored_result_thread_partition(&d_bucket, d_start_index, d_end_index, &d_last_bucket_index, &d_result_last_index, number_of_parts, &d_thread_start_index, &d_thread_end_index);
//	//for timing
//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[3]=cuda_elapsedTime;
//	//end for timing


//	 cudaEventRecord(cuda_start, 0);//for timing
	//4. Save result
	save_result<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(d_bucket), rpc(*d_start_index), rpc(*d_end_index), rpc(d_last_bucket_index), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(*d_result));
	tailored_update_result_last_position<<<number_of_parts, 1>>> (rpc(d_result_last_index), rpc(d_thread_end_index),THREADS_PER_BLOCK);
//	//for timing
//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[4]=cuda_elapsedTime;
//	//end for timing

//	 cudaEventRecord(cuda_start, 0);//for timing
	//5. Construct next round
	int next_size = d_end_index_of_each_part[number_of_parts-1];
	int miss_size = reduce(d_miss.begin(), d_miss.end());
	device_vector<int> d_next_round(next_size);
	tailored_miss_thread_parition(&d_bucket, d_start_index, d_end_index, &d_last_bucket_index, &d_end_index_of_each_part, number_of_parts,&d_thread_start_index,  &d_thread_end_index);
	save_miss_array<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(d_bucket), rpc(*d_start_index), rpc(*d_end_index), rpc(d_last_bucket_index), rpc(d_thread_start_index),  rpc(d_thread_end_index), rpc(d_next_round));
//	//for timing
//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[5]=cuda_elapsedTime;
//	//end for timing

//	 cudaEventRecord(cuda_start, 0);//for timing
	//6. Update min max array
	update_min_max<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(*d_min), rpc(*d_max), rpc(d_last_bucket_index), number_of_buckets);
//	//for timing
//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[6]=cuda_elapsedTime;
//	//end for timing
	//========================================================================================================================

	//Loop until find all k
	//========================================================================================================================
	while(miss_size > 0)
	{
		number_of_buckets = BUCKER_NUM_PER_PART < next_size/2 ? BUCKER_NUM_PER_PART : next_size/2+1;

//		 cudaEventRecord(cuda_start, 0);//for timing
		//1. Assign bucket index
		d_bucket.resize(next_size);
		//d_bucket_size.resize(number_of_parts*number_of_buckets);
		thrust::fill(d_bucket_size.begin(), d_bucket_size.end(), 0);
		reassign_bucket<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(*d_data), sel, rpc(d_next_round), rpc(d_bucket), rpc(d_end_index_of_each_part), rpc(d_bucket_size), rpc(*d_min), rpc(*d_max), number_of_buckets);
//		//for timing
//		cudaEventRecord(cuda_stop, 0);//for timing
//		cudaEventSynchronize(cuda_stop);
//		cuda_elapsedTime=0;
//		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//		timeRec[7]+=cuda_elapsedTime;
//		//end for timing

//		 cudaEventRecord(cuda_start, 0);//for timing
		//2. Count last index and miss
		device_vector<int> d_new_miss(number_of_parts);
		device_vector<int> d_new_end_index_of_each_part(number_of_parts);
		tailored_count_sum(&d_miss, &d_bucket_size, number_of_parts, number_of_buckets, &d_last_bucket_index, &d_new_miss, &d_new_end_index_of_each_part);
//		//for timing
//		cudaEventRecord(cuda_stop, 0);//for timing
//		cudaEventSynchronize(cuda_stop);
//		cuda_elapsedTime=0;
//		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//		timeRec[8]+=cuda_elapsedTime;
//		//end for timing


//		 cudaEventRecord(cuda_start, 0);//for timing
		//3. Tid partition
		tailored_result_thread_partition(&d_bucket, &d_end_index_of_each_part, &d_last_bucket_index, &d_new_miss, d_min, d_max, &d_result_last_index, number_of_parts,&d_thread_start_index, &d_thread_end_index);
//		//for timing
//		cudaEventRecord(cuda_stop, 0);//for timing
//		cudaEventSynchronize(cuda_stop);
//		cuda_elapsedTime=0;
//		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//		timeRec[9]+=cuda_elapsedTime;
//		//end for timing

//		 cudaEventRecord(cuda_start, 0);//for timing
		//4. Save result
		save_result<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(d_bucket), rpc(d_end_index_of_each_part), rpc(d_last_bucket_index), rpc(*d_min), rpc(*d_max), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(*d_result));
		tailored_update_result_last_position<<<number_of_parts, 1 >>>
				(rpc(d_result_last_index), rpc(d_thread_end_index),THREADS_PER_BLOCK);
//		//for timing
//		cudaEventRecord(cuda_stop, 0);//for timing
//		cudaEventSynchronize(cuda_stop);
//		cuda_elapsedTime=0;
//		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//		timeRec[10]+=cuda_elapsedTime;
//		//end for timing

//		 cudaEventRecord(cuda_start, 0);//for timing
		//5. Construct next round
		int new_next_size = d_new_end_index_of_each_part[number_of_parts-1];
		int new_miss_size = reduce(d_new_miss.begin(), d_new_miss.end());
		tailored_miss_thread_parition(&d_bucket, &d_end_index_of_each_part, &d_last_bucket_index, &d_new_end_index_of_each_part, number_of_parts,&d_thread_start_index,  &d_thread_end_index);
		d_next_round.resize(new_next_size);
		save_miss_array<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(d_bucket), rpc(d_end_index_of_each_part), rpc(d_last_bucket_index), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(d_next_round));

		update_min_max<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(*d_min), rpc(*d_max), rpc(d_last_bucket_index), number_of_buckets);
//		//for timing
//		cudaEventRecord(cuda_stop, 0);//for timing
//		cudaEventSynchronize(cuda_stop);
//		cuda_elapsedTime=0;
//		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//		timeRec[11]+=cuda_elapsedTime;
//		//end for timing

		d_miss = d_new_miss;
		d_end_index_of_each_part = d_new_end_index_of_each_part;

		miss_size = new_miss_size;
		next_size = new_next_size;
	}

//	//for timing
//	for(int i=0;i<12;i++){
//		cout<<" for timing: bucket topk step ["<<i<<"] time:{"<<timeRec[i]<<"} "<<endl;
//	}
//	//end for timing

}





template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, device_vector<float> *d_min, device_vector<float> *d_max, device_vector<int> *d_k, device_vector<int> *d_start_index, device_vector<int> *d_end_index, int number_of_parts, device_vector<int> *d_result)
{
	//for timing
	cudaProfilerStart();
	cudaEvent_t cuda_start;
	cudaEventCreate(&cuda_start);
	cudaEvent_t cuda_stop;
	cudaEventCreate(&cuda_stop);
	float timeRec[12];
	for(int ti=0;ti<12;ti++){timeRec[ti]=0;}
	float cuda_elapsedTime=0;
	//for timing

	cudaEventRecord(cuda_start, 0);//for timing
	//========================================================================================================================
	device_vector<int> d_result_end_index(number_of_parts);
	device_vector<int> d_result_last_index(number_of_parts);
	inclusive_scan((*d_k).begin(), (*d_k).end(), d_result_end_index.begin());
	exclusive_scan((*d_k).begin(), (*d_k).end(), d_result_last_index.begin());

	const int size = (*d_end_index)[number_of_parts-1] - (*d_start_index)[0];
	//cout<<"bucket_topk d_bucket size:"<<size<<" mem:"<<size*sizeof(Bucket)<<" d_data size:"<<(*d_data).size()<<" mem:"<<(*d_data).size()*sizeof((*d_data)[0])<<" total mem:"<< size*sizeof(Bucket) +(*d_data).size()*sizeof((*d_data)[0]) <<endl;

	const int result_size = d_result_end_index[number_of_parts-1];
	int number_of_buckets = BUCKER_NUM_PER_PART < size/2 ? BUCKER_NUM_PER_PART : size/2+1;

	(*d_result).resize(result_size);
	thrust::fill((*d_result).begin(), (*d_result).end(), -1);
	//========================================================================================================================
	 cudaEventRecord(cuda_stop, 0);//for timing
	 cudaEventSynchronize(cuda_stop);
	 cuda_elapsedTime=0;
	 cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	 timeRec[0]=cuda_elapsedTime;

	//First Round
	//========================================================================================================================
	 cudaEventRecord(cuda_start, 0);//for timing
	//1. Assign bucket index
	device_vector<Bucket> d_bucket(size);
	device_vector<int> d_bucket_size(number_of_parts*number_of_buckets);
	assign_bucket<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(*d_data), sel, rpc(d_bucket), rpc(*d_start_index), rpc(*d_end_index), rpc(d_bucket_size), rpc(*d_min), rpc(*d_max), number_of_buckets);
	//for timing
	cudaEventRecord(cuda_stop, 0);//for timing
	cudaEventSynchronize(cuda_stop);
	cuda_elapsedTime=0;
	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	timeRec[1]=cuda_elapsedTime;
	//end for timing

	 cudaEventRecord(cuda_start, 0);//for timing
	//2. Count last index and miss
	device_vector<int> d_last_bucket_index(number_of_parts);
	device_vector<int> d_miss(number_of_parts);
	device_vector<int> d_end_index_of_each_part(number_of_parts);
	count_sum(d_k, &d_bucket_size, number_of_parts, number_of_buckets, &d_last_bucket_index, &d_miss, &d_end_index_of_each_part);
	//for timing
	cudaEventRecord(cuda_stop, 0);//for timing
	cudaEventSynchronize(cuda_stop);
	cuda_elapsedTime=0;
	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	timeRec[2]=cuda_elapsedTime;
	//end for timing

	 cudaEventRecord(cuda_start, 0);//for timing
	//3. Tid partition
	device_vector<int> d_thread_start_index(number_of_parts*THREADS_PER_BLOCK);
	device_vector<int> d_thread_end_index(number_of_parts*THREADS_PER_BLOCK);
	result_thread_partition(&d_bucket, d_start_index, d_end_index, &d_last_bucket_index, &d_result_last_index, number_of_parts, &d_thread_start_index, &d_thread_end_index);
	//for timing
	cudaEventRecord(cuda_stop, 0);//for timing
	cudaEventSynchronize(cuda_stop);
	cuda_elapsedTime=0;
	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	timeRec[3]=cuda_elapsedTime;
	//end for timing


	 cudaEventRecord(cuda_start, 0);//for timing
	//4. Save result
	save_result<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(d_bucket), rpc(*d_start_index), rpc(*d_end_index), rpc(d_last_bucket_index), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(*d_result));
	update_result_last_position<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(d_result_last_index), rpc(d_thread_end_index));
	//for timing
	cudaEventRecord(cuda_stop, 0);//for timing
	cudaEventSynchronize(cuda_stop);
	cuda_elapsedTime=0;
	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	timeRec[4]=cuda_elapsedTime;
	//end for timing

	 cudaEventRecord(cuda_start, 0);//for timing
	//5. Construct next round
	int next_size = d_end_index_of_each_part[number_of_parts-1];
	int miss_size = reduce(d_miss.begin(), d_miss.end());
	device_vector<int> d_next_round(next_size);
	miss_thread_parition(&d_bucket, d_start_index, d_end_index, &d_last_bucket_index, &d_end_index_of_each_part, number_of_parts, &d_thread_start_index, &d_thread_end_index);
	save_miss_array<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(d_bucket), rpc(*d_start_index), rpc(*d_end_index), rpc(d_last_bucket_index), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(d_next_round));
	//for timing
	cudaEventRecord(cuda_stop, 0);//for timing
	cudaEventSynchronize(cuda_stop);
	cuda_elapsedTime=0;
	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	timeRec[5]=cuda_elapsedTime;
	//end for timing

	 cudaEventRecord(cuda_start, 0);//for timing
	//6. Update min max array
	update_min_max<<<number_of_parts, THREADS_PER_BLOCK>>>
			(rpc(*d_min), rpc(*d_max), rpc(d_last_bucket_index), number_of_buckets);
	//for timing
	cudaEventRecord(cuda_stop, 0);//for timing
	cudaEventSynchronize(cuda_stop);
	cuda_elapsedTime=0;
	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
	timeRec[6]=cuda_elapsedTime;
	//end for timing
	//========================================================================================================================

	//Loop until find all k
	//========================================================================================================================
	while(miss_size > 0)
	{
		number_of_buckets = BUCKER_NUM_PER_PART < next_size/2 ? BUCKER_NUM_PER_PART : next_size/2+1;

		 cudaEventRecord(cuda_start, 0);//for timing
		//1. Assign bucket index
		d_bucket.resize(next_size);
		d_bucket_size.resize(number_of_parts*number_of_buckets);
		thrust::fill(d_bucket_size.begin(), d_bucket_size.end(), 0);
		reassign_bucket<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(*d_data), sel, rpc(d_next_round), rpc(d_bucket), rpc(d_end_index_of_each_part), rpc(d_bucket_size), rpc(*d_min), rpc(*d_max), number_of_buckets);
		//for timing
		cudaEventRecord(cuda_stop, 0);//for timing
		cudaEventSynchronize(cuda_stop);
		cuda_elapsedTime=0;
		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
		timeRec[7]+=cuda_elapsedTime;
		//end for timing

		 cudaEventRecord(cuda_start, 0);//for timing
		//2. Count last index and miss
		device_vector<int> d_new_miss(number_of_parts);
		device_vector<int> d_new_end_index_of_each_part(number_of_parts);
		count_sum(&d_miss, &d_bucket_size, number_of_parts, number_of_buckets, &d_last_bucket_index, &d_new_miss, &d_new_end_index_of_each_part);
		//for timing
		cudaEventRecord(cuda_stop, 0);//for timing
		cudaEventSynchronize(cuda_stop);
		cuda_elapsedTime=0;
		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
		timeRec[8]+=cuda_elapsedTime;
		//end for timing


		 cudaEventRecord(cuda_start, 0);//for timing
		//3. Tid partition
		result_thread_partition(&d_bucket, &d_end_index_of_each_part, &d_last_bucket_index, &d_new_miss, d_min, d_max, &d_result_last_index, number_of_parts, &d_thread_start_index, &d_thread_end_index);
		//for timing
		cudaEventRecord(cuda_stop, 0);//for timing
		cudaEventSynchronize(cuda_stop);
		cuda_elapsedTime=0;
		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
		timeRec[9]+=cuda_elapsedTime;
		//end for timing

		 cudaEventRecord(cuda_start, 0);//for timing
		//4. Save result
		save_result<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(d_bucket), rpc(d_end_index_of_each_part), rpc(d_last_bucket_index), rpc(*d_min), rpc(*d_max), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(*d_result));
		update_result_last_position<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(d_result_last_index), rpc(d_thread_end_index));
		//for timing
		cudaEventRecord(cuda_stop, 0);//for timing
		cudaEventSynchronize(cuda_stop);
		cuda_elapsedTime=0;
		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
		timeRec[10]+=cuda_elapsedTime;
		//end for timing

		 cudaEventRecord(cuda_start, 0);//for timing
		//5. Construct next round
		int new_next_size = d_new_end_index_of_each_part[number_of_parts-1];
		int new_miss_size = reduce(d_new_miss.begin(), d_new_miss.end());
		miss_thread_parition(&d_bucket, &d_end_index_of_each_part, &d_last_bucket_index, &d_new_end_index_of_each_part, number_of_parts, &d_thread_start_index, &d_thread_end_index);
		d_next_round.resize(new_next_size);
		save_miss_array<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(d_bucket), rpc(d_end_index_of_each_part), rpc(d_last_bucket_index), rpc(d_thread_start_index), rpc(d_thread_end_index), rpc(d_next_round));

		update_min_max<<<number_of_parts, THREADS_PER_BLOCK>>>
				(rpc(*d_min), rpc(*d_max), rpc(d_last_bucket_index), number_of_buckets);
		//for timing
		cudaEventRecord(cuda_stop, 0);//for timing
		cudaEventSynchronize(cuda_stop);
		cuda_elapsedTime=0;
		cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
		timeRec[11]+=cuda_elapsedTime;
		//end for timing

		d_miss = d_new_miss;
		d_end_index_of_each_part = d_new_end_index_of_each_part;

		miss_size = new_miss_size;
		next_size = new_next_size;
	}

	//for timing
	for(int i=0;i<12;i++){
		cout<<" bucket topk step ["<<i<<"] time:{"<<timeRec[i]<<"} "<<endl;
	}
	//end for timing

}

/* CONSTANT MIN MAX */
template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, float MIN, float MAX, device_vector<int> *d_k, device_vector<int> *d_start_index, device_vector<int> *d_end_index, int number_of_parts, device_vector<int> *d_result)
{
	device_vector<float> d_min(number_of_parts, MIN), d_max(number_of_parts, MAX);
	bucket_topk(d_data, sel, &d_min, &d_max, d_k, d_start_index, d_end_index, number_of_parts, d_result);
}

/**********************************************************************************************************/

template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, device_vector<float> *d_min, device_vector<float> *d_max, device_vector<int> *d_k, device_vector<int> *d_end_index, int number_of_parts, device_vector<int> *d_result)
{
	device_vector<int> d_start_index(number_of_parts);
	default_start_index<<<number_of_parts, THREADS_PER_BLOCK>>>(rpc(d_start_index), rpc(*d_end_index));
	bucket_topk(d_data, sel, d_min, d_max, d_k, &d_start_index, d_end_index, number_of_parts, d_result);
}

/* CONSTANT MIN MAX */
template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, float MIN, float MAX, device_vector<int> *d_k, device_vector<int> *d_end_index, int number_of_parts, device_vector<int> *d_result)
{
	device_vector<float> d_min(number_of_parts, MIN), d_max(number_of_parts, MAX);
	bucket_topk(d_data, sel, &d_min, &d_max, d_k, d_end_index, number_of_parts, d_result);
}

/**********************************************************************************************************/

template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, device_vector<float> *d_min, device_vector<float> *d_max, device_vector<int> *d_k, device_vector<int> *d_start_index, device_vector<int> *d_end_index, int number_of_parts)
{
	device_vector<int> d_result;
	device_vector<int> d_result_end_index(number_of_parts);
	inclusive_scan((*d_k).begin(), (*d_k).end(), d_result_end_index.begin());
	bucket_topk(d_data, sel, d_min, d_max, d_k, d_start_index, d_end_index, number_of_parts, &d_result);
	shift_result_for<<<number_of_parts, THREADS_PER_BLOCK, BUCKET_TOPK_MAX_K*sizeof(int)>>>
			(rpc(*d_data), rpc(*d_start_index), rpc(d_result), Bucket_Topk_ValueOfInt(), rpc(d_result_end_index));
}

/* CONSTANT MIN MAX */
template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, float MIN, float MAX, device_vector<int> *d_k, device_vector<int> *d_start_index, device_vector<int> *d_end_index, int number_of_parts)
{
	device_vector<float> d_min(number_of_parts, MIN), d_max(number_of_parts, MAX);
	bucket_topk(d_data, sel, &d_min, &d_max, d_k, d_start_index, d_end_index, number_of_parts);
}

/**********************************************************************************************************/

template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, device_vector<float> *d_min, device_vector<float> *d_max, device_vector<int> *d_k, device_vector<int> *d_end_index, int number_of_parts)
{
	device_vector<int> d_start_index(number_of_parts);
	default_start_index<<<number_of_parts, THREADS_PER_BLOCK>>>(rpc(d_start_index), rpc(*d_end_index));
	bucket_topk(d_data, sel, d_min, d_max, d_k, &d_start_index, d_end_index, number_of_parts);
}


/* CONSTANT MIN MAX */
template<typename Data, typename GetValue>
__host__ void bucket_topk(device_vector<Data> *d_data, GetValue sel, float MIN, float MAX, device_vector<int> *d_k, device_vector<int> *d_end_index, int number_of_parts)
{
	device_vector<float> d_min(number_of_parts, MIN), d_max(number_of_parts, MAX);
	bucket_topk(d_data, sel, &d_min, &d_max, d_k, d_end_index, number_of_parts);
}
#endif
