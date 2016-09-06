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

#include "multithreading.h"

pthread_t cutStartThread(ThreadFunc func, void *data) {
  pthread_t thread;
  // thread attributes is null
  pthread_create(&thread, NULL, func, data);
  return thread;
}

void cutEndThread(pthread_t thread) {
  // not gathering temination state of specified thread
  pthread_join(thread, NULL);
}

void cutDestroyThread(pthread_t thread) {
  pthread_cancel(thread);
}

void cutWaitForThreads(const pthread_t *threads, int num) {
  for (int i = 0; i < num; i++) {
    cutEndThread(threads[i]);
  }
}

void cutCreateBarrier(int release_count, CUTBarrier *bar) {
  bar->current_count = 0;
  bar->release_count = release_count;
  pthread_mutex_init(&bar->mutex, 0);
  pthread_cond_init(&bar->condition, 0);
}

bool cutIncrementBarrierAndWaitForBroadcast(CUTBarrier *barrier, bool reset) {
  bool result = false;

  pthread_mutex_lock(&barrier->mutex);
  ++barrier->current_count;

  if (barrier->current_count == barrier->release_count) {
    // this is the last thread of the 'release_count' threads
    result = true;
    if (reset) barrier->current_count = 0;
    pthread_mutex_unlock(&barrier->mutex);

    pthread_cond_broadcast(&barrier->condition);
  } else {  // barrier->current_count < barrier->release_count
    // No need of 'while' because we use broadcast here.
    // We must make sure that when a thread release the mutex, it is already
    // waiting on the condition!
    pthread_cond_wait(&barrier->condition, &barrier->mutex);
    pthread_mutex_unlock(&barrier->mutex);
  }
  return result;
}

bool cutIncrementBarrierNoWait(CUTBarrier *barrier) {
  int my_barrier_count = 0;

  pthread_mutex_lock(&barrier->mutex);
  my_barrier_count = ++barrier->current_count;
  pthread_mutex_unlock(&barrier->mutex);

  if (my_barrier_count == barrier->release_count) {
    pthread_cond_broadcast(&barrier->condition);
    return true;
  }
  return false;
}

void cutWaitForBarrier(CUTBarrier *barrier) {
  pthread_mutex_lock(&barrier->mutex);
  if (barrier->current_count < barrier->release_count) {
    // No need of 'while' because we use broadcast here.
    pthread_cond_wait(&barrier->condition, &barrier->mutex);
  }
  pthread_mutex_unlock(&barrier->mutex);
}

void cutDestroyBarrier(CUTBarrier *barrier) {
  pthread_mutex_destroy(&barrier->mutex);
  pthread_cond_destroy(&barrier->condition);
}
