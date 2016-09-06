// *****************************************************************************
// Filename:    device_util.h
// Date:        2012-12-11 14:26
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef DEVICE_UTIL_H_
#define DEVICE_UTIL_H_

#define GET_NUM_THREAD_TID() \
    const unsigned int num_threads = gridDim.x * blockDim.x; \
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x

#define GET_NUM_THREAD_TID_AND_RETURN_IF_TID_GE(NUM) \
    GET_NUM_THREAD_TID(); \
    if (tid >= NUM) return

#define ALLOCATE_ON_DEVICE(TYPE, PTR, SIZE) \
    checkCudaErrors(cudaMalloc(&PTR, SIZE * sizeof(TYPE)))

#define DEALLOCATE_ON_DEVICE(PTR) \
    checkCudaErrors(cudaFree(PTR)); \
    PTR = NULL

#define INIT_OUT_MEMBERS(TYPE, MEMBER, SIZE, VALUE) { \
    thrust::device_ptr<TYPE> thr_member(MEMBER); \
    thrust::fill(thr_member, thr_member + SIZE, VALUE); \
  }

// We use '{ }' here because we need to define local members and to protect them
// from naming conflicts.
#define SHUFFLE_MEMBER(TYPE, MEMBER, SIZE, BUF, THR_INDEXMAP) { \
    thrust::device_ptr<TYPE> thr_member(MEMBER); \
    thrust::device_ptr<TYPE> thr_buf((TYPE*)(BUF)); \
    thrust::copy(thr_member, thr_member + SIZE, thr_buf); \
    thrust::gather(THR_INDEXMAP, THR_INDEXMAP + SIZE, thr_buf, thr_member); \
  }

#endif
