#ifndef COUNT_SUM_INC
#define COUNT_SUM_INC

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
using namespace thrust;

static void tailored_count_sum(int k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part);
static __global__ void tailored_check_index(int k, int* bucket_size_sum, int number_of_buckets, int* last_bucket_index, int* miss);
static void thrust_copy_size(device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets,
		device_vector<int> *last_bucket_index,  device_vector<int> *end_index_of_each_part);
static __global__ void tailored_copy_size(int* last_bucket_index, int* bucket_size_endIdx, int number_of_buckets, int* new_partition_size);

static void count_sum(device_vector<int> *k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part);
static __global__ void copy_size(int* last_bucket_index, int* bucket_size, int number_of_buckets, int* new_partition_size);
static __global__ void check_index(int* k, int* bucket_size_sum, int number_of_buckets, int* last_bucket_index, int* miss);


static void tailored_count_sum(int k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part)
{
//	//for timing
//		cudaProfilerStart();
//		cudaEvent_t cuda_start;
//		cudaEventCreate(&cuda_start);
//		cudaEvent_t cuda_stop;
//		cudaEventCreate(&cuda_stop);
//		float timeRec[12];
//		for(int ti=0;ti<12;ti++){timeRec[ti]=0;}
//		float cuda_elapsedTime=0;
//		//for timing


//	cudaEventRecord(cuda_start, 0);//for timing

	//device_vector<int> d_bucket_size_sum(number_of_buckets*number_of_parts);
	//inclusive_scan((*d_bucket_size).begin(), (*d_bucket_size).end(), d_bucket_size_sum.begin());
	inclusive_scan((*d_bucket_size).begin(), (*d_bucket_size).end(), (*d_bucket_size).begin());//re-use bucket_size


//	cudaEventRecord(cuda_stop, 0);//for timing
//	 cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	 cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	 timeRec[1]=cuda_elapsedTime;

//	 cudaEventRecord(cuda_start, 0);//for timing

	tailored_check_index<<<number_of_parts, THREADS_PER_BLOCK>>>(
			k,
			raw_pointer_cast((*d_bucket_size).data()),//re-use bucket_size to replace d_bucket_size_sum
			number_of_buckets,
			raw_pointer_cast((*last_bucket_index).data()),
			raw_pointer_cast((*miss).data()));

//	cudaEventRecord(cuda_stop, 0);//for timing
//		 cudaEventSynchronize(cuda_stop);
//		cuda_elapsedTime=0;
//		 cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//		 timeRec[2]=cuda_elapsedTime;


//	cudaEventRecord(cuda_start, 0);//for timing

	tailored_copy_size<<<number_of_parts, 1>>>(raw_pointer_cast((*last_bucket_index).data()), raw_pointer_cast((*d_bucket_size).data()), number_of_buckets, raw_pointer_cast((*end_index_of_each_part).data()));


//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[3]=cuda_elapsedTime;

//	cudaEventRecord(cuda_start, 0);//for timing

	inclusive_scan((*end_index_of_each_part).begin(), (*end_index_of_each_part).end(), (*end_index_of_each_part).begin());

//	cudaEventRecord(cuda_stop, 0);//for timing
//	cudaEventSynchronize(cuda_stop);
//	cuda_elapsedTime=0;
//	cudaEventElapsedTime(&cuda_elapsedTime, cuda_start, cuda_stop);
//	timeRec[4]=cuda_elapsedTime;

//	//for timing
//		for(int i=0;i<5;i++){
//			cout<<" tailored_count_sum step ["<<i<<"] time:{"<<timeRec[i]<<"} "<<endl;
//		}
//		//end for timing
}


static __global__ void tailored_check_index(int k, int* bucket_size_sum, int number_of_buckets, int* last_bucket_index, int* miss)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int blk_start_index = bid*number_of_buckets;
	int blk_end_index = (bid+1)*number_of_buckets;
	int blk_before_end_index = blk_start_index-1;
	int size_before = blk_before_end_index<0 ? 0 : bucket_size_sum[blk_before_end_index];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(index==blk_start_index)
			{
				if(bucket_size_sum[index] - size_before > k)
				{
					last_bucket_index[bid] = index;
					miss[bid] = k;
				}
			}
			else if(bucket_size_sum[index] - size_before > k && bucket_size_sum[index-1] - size_before <= k)
			{
					last_bucket_index[bid] = index;
					miss[bid] = k - (bucket_size_sum[index-1]- size_before);
			}
			else if(index == blk_end_index-1 && bucket_size_sum[index] - size_before <= k)
			{
				last_bucket_index[bid] = blk_end_index;
				miss[bid] = 0;
			}
		}
	}
}


static void count_sum(device_vector<int> *k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part)
{
	device_vector<int> d_bucket_size_sum(number_of_buckets*number_of_parts);
	inclusive_scan((*d_bucket_size).begin(), (*d_bucket_size).end(), d_bucket_size_sum.begin());
	check_index<<<number_of_parts, THREADS_PER_BLOCK>>>(
			raw_pointer_cast((*k).data()),
			raw_pointer_cast(d_bucket_size_sum.data()),
			number_of_buckets,
			raw_pointer_cast((*last_bucket_index).data()),
			raw_pointer_cast((*miss).data()));

	copy_size<<<number_of_parts, 1>>>(raw_pointer_cast((*last_bucket_index).data()), raw_pointer_cast((*d_bucket_size).data()), number_of_buckets, raw_pointer_cast((*end_index_of_each_part).data()));
	inclusive_scan((*end_index_of_each_part).begin(), (*end_index_of_each_part).end(), (*end_index_of_each_part).begin());
}

static void tailored_count_sum(device_vector<int> *k, device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets, device_vector<int> *last_bucket_index, device_vector<int> *miss, device_vector<int> *end_index_of_each_part)
{
	//device_vector<int> d_bucket_size_sum(number_of_buckets*number_of_parts);
	inclusive_scan((*d_bucket_size).begin(), (*d_bucket_size).end(), (*d_bucket_size).begin());//to re-use d_bucket_size
	check_index<<<number_of_parts, THREADS_PER_BLOCK>>>(
			raw_pointer_cast((*k).data()),
			raw_pointer_cast((*d_bucket_size).data()),//to re-use d_bucket_size
			number_of_buckets,
			raw_pointer_cast((*last_bucket_index).data()),
			raw_pointer_cast((*miss).data()));

	tailored_copy_size<<<number_of_parts, 1>>>(raw_pointer_cast((*last_bucket_index).data()), raw_pointer_cast((*d_bucket_size).data()), number_of_buckets, raw_pointer_cast((*end_index_of_each_part).data()));
	inclusive_scan((*end_index_of_each_part).begin(), (*end_index_of_each_part).end(), (*end_index_of_each_part).begin());
}



static __global__ void check_index(int* k, int* bucket_size_sum, int number_of_buckets, int* last_bucket_index, int* miss)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	int blk_start_index = bid*number_of_buckets;
	int blk_end_index = (bid+1)*number_of_buckets;
	int blk_before_end_index = blk_start_index-1;
	int size_before = blk_before_end_index<0 ? 0 : bucket_size_sum[blk_before_end_index];

	int round = (blk_end_index-blk_start_index)/blockDim.x + 1;
	int offset = blockDim.x;

	int index;
	for(int i=0; i<round; i++)
	{
		index = (blk_start_index+tid)+i*offset;
		if(index < blk_end_index)
		{
			if(tid==0 && i==0)
			{
				if(bucket_size_sum[index] - size_before > k[bid])
				{
					last_bucket_index[bid] = index;
					miss[bid] = k[bid];
				}
			}
			else if(bucket_size_sum[index] - size_before > k[bid] && bucket_size_sum[index-1] - size_before <= k[bid])
			{
					last_bucket_index[bid] = index;
					miss[bid] = k[bid] - (bucket_size_sum[index-1]- size_before);
			}
			else if(index == blk_end_index-1 && bucket_size_sum[index] - size_before <= k[bid])
			{
				last_bucket_index[bid] = blk_end_index;
				miss[bid] = 0;
			}
		}
	}
}

static __global__ void copy_size(int* last_bucket_index, int* bucket_size, int number_of_buckets, int* new_partition_size)
{
	int bid = blockIdx.x;
	int last = (bid+1)*number_of_buckets;
	if(last_bucket_index[bid] < last)
		new_partition_size[bid] = bucket_size[last_bucket_index[bid]];
	else
		new_partition_size[bid] = 0;
}

static __global__ void tailored_copy_size(int* last_bucket_index, int* bucket_size_endIdx, int number_of_buckets, int* new_partition_size)
{
	int bid = blockIdx.x;
	int last = (bid+1)*number_of_buckets;
	if(last_bucket_index[bid] < last){
		int bucket_size_startIdx= (last_bucket_index[bid]==0)?0: bucket_size_endIdx[last_bucket_index[bid]-1];
		new_partition_size[bid] = bucket_size_endIdx[last_bucket_index[bid]]-bucket_size_startIdx;
	}
	else {
		new_partition_size[bid] = 0;
	}
}


struct copy_last_index{
	int number_of_buckets;

	__host__ __device__
	copy_last_index(int number_of_buckets){
		this->number_of_buckets=number_of_buckets;
	}

	template <class Tuple>
	__host__ __device__
	void operator()(Tuple t)
	{

		int last_bucket_index = thrust::get<1>(t);
		int bid = thrust::get<2>(t);

		if(last_bucket_index<(bid+1)*number_of_buckets){
			thrust::get<3>(t) =thrust::get<0>(t);
		}else{

			thrust::get<3>(t) =0;
		}

	}
};

static void thrust_copy_size(device_vector<int> *d_bucket_size, int number_of_parts, int number_of_buckets,
		device_vector<int> *last_bucket_index,  device_vector<int> *end_index_of_each_part)
{
	thrust::counting_iterator<int> countIt_begin(0);
	thrust::counting_iterator<int> countIt_end = countIt_begin
			+ number_of_parts;

	thrust::for_each(
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator((*d_bucket_size).begin(),
									(*last_bucket_index).begin()),
							(*last_bucket_index).begin(), countIt_begin,
							(*end_index_of_each_part).begin())),
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator((*d_bucket_size).begin(),
									(*last_bucket_index).end()),
							(*last_bucket_index).end(), countIt_end,
							(*end_index_of_each_part).end())),
			copy_last_index(number_of_buckets));
}
#endif
