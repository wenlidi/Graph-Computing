/*
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef MULTITHREADING_H
#define MULTITHREADING_H

// Simple portable thread library.
// POSIX threads.
#include <pthread.h>

typedef void *(*ThreadFunc)(void *);

struct CUTBarrier {
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  int release_count;
  int current_count;
};

// #ifdef __cplusplus
// extern "C" {
// #endif

// Create thread.
pthread_t cutStartThread(ThreadFunc func, void *data);

// Wait for thread to finish.
void cutEndThread(pthread_t thread);

// Destroy thread.
void cutDestroyThread(pthread_t thread);

// Wait for multiple threads.
void cutWaitForThreads(const pthread_t *threads, int num);

// Create barrier.
void cutCreateBarrier(int release_count, CUTBarrier *bar);

// If @reset is true:
// 'release_count' threads waiting for all of them to finish: each except the
// last thread waiting on all other threads to finish; when the last thread
// finishes, it broadcast the signal and reset 'current_count = 0'.
//
// Else if @reset is false:
// 'release_count' + n (an arbitary number) threads waiting for those
// 'release_count' threads to finish: each except the last thread of the
// 'release_count' threads increase the count and wait fot the last one to
// broadcast. Those n threads just invoke the following function
// cutWaitForBarrier to wait.
//
// Returns true if the invocation did a broadcast.
//
// NOTE: Only allowing exactly 'release_count' threads invoke this function!
bool cutIncrementBarrierAndWaitForBroadcast(CUTBarrier *barrier, bool reset);

// n threads waiting for 'release_count' threads to finish. This is similar to
// the previous function cutIncrementBarrierAndWaitForBroadcast with
// reset = false, except that those 'release_count' threads will not be
// blocked when invoking this function.
//
// Returns true if this is the last thread of the 'release_count' threads that
// invokes this function.
//
// NOTE: Only allowing exactly 'release_count' threads invoke this function!
bool cutIncrementBarrierNoWait(CUTBarrier *barrier);

void cutWaitForBarrier(CUTBarrier *barrier);

// Destory barrier
void cutDestroyBarrier(CUTBarrier *barrier);

// #ifdef __cplusplus
// } // extern "C"
// #endif

#endif // MULTITHREADING_H
