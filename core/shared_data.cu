// *****************************************************************************
// Filename:    shared_data.cc
// Date:        2012-12-12 14:07
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "shared_data.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "multithreading.h"
#include "hash_types.h"
#include "host_hash_functions.h"
#include "host_graph.h"

using std::cerr;
using std::cin;
using std::cout;
using std::endl;

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> [SharedData]: "
#endif

SharedData::SharedData(const Config *config)
    : conf(config),
      // Barriers
      global_barrier(),
      vcon_memory_barrier(),
      vcon_barrier(),
      econ_memory_barrier(),
      econ_barrier(),
      all_control_threads_syn_barrier(),
      superstep_end_barrier(),
      // Global
      global(),
      global_set(false),
      // vcon related
      vcon_size(NULL),
      vcon_buf(NULL),
      vcon_buf_mutex(NULL),
      vcon_buf_full_cond(NULL),
      vcon_buf_empty_cond(NULL),
      vcon_read_finished(false),
      // econ related
      econ_size(NULL),
      econ_buf(NULL),
      econ_buf_mutex(NULL),
      econ_buf_full_cond(NULL),
      econ_buf_empty_cond(NULL),
      econ_read_finished(false),
      // Other member
      in_edges_per_gpu(NULL),
      out_edges_per_gpu(NULL),
      gpu_status(),
      gpu_status_barrier_count(0),
      gpu_status_mutex(),
      gpu_status_condition(),
      // For global shuffle
      mcon_recv_ptrs(NULL),
      mcon_recv_ptrs_mutex(NULL),
      mcon_recv_used(NULL),
      // For perfromance test
      host_graph_mutex(),
      host_graph(config) {
  const unsigned int num_reading_threads = conf->GetNumReadingThreads();
  const unsigned int num_gpu_control_threads = conf->GetNumGPUControlThreads();

  // Init barriers.
  cutCreateBarrier(1, &global_barrier);
  cutCreateBarrier(num_gpu_control_threads, &vcon_memory_barrier);
  cutCreateBarrier(num_reading_threads, &vcon_barrier);
  cutCreateBarrier(num_gpu_control_threads, &econ_memory_barrier);
  cutCreateBarrier(num_reading_threads, &econ_barrier);
  cutCreateBarrier(num_gpu_control_threads, &all_control_threads_syn_barrier);
  cutCreateBarrier(num_gpu_control_threads, &superstep_end_barrier);

  // Init vcon buffer and corresponding control parameters.
  vcon_size = new unsigned int[num_gpu_control_threads];
  vcon_buf = new HostInVertexContent[num_gpu_control_threads];
  vcon_buf_mutex = new pthread_mutex_t[num_gpu_control_threads];
  vcon_buf_full_cond = new pthread_cond_t[num_gpu_control_threads];
  vcon_buf_empty_cond = new pthread_cond_t[num_gpu_control_threads];
  vcon_read_finished = false;
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    vcon_size[i] = 0;
    vcon_buf[i].Allocate(kBufSize);
    pthread_mutex_init(&vcon_buf_mutex[i], 0);
    pthread_cond_init(&vcon_buf_full_cond[i], 0);
    pthread_cond_init(&vcon_buf_empty_cond[i], 0);
  }

  // Init econ buffer and corresponding control parameters.
  econ_size = new unsigned int[num_gpu_control_threads];
  econ_buf = new HostInEdgeContent[num_gpu_control_threads];
  econ_buf_mutex = new pthread_mutex_t[num_gpu_control_threads];
  econ_buf_full_cond = new pthread_cond_t[num_gpu_control_threads];
  econ_buf_empty_cond = new pthread_cond_t[num_gpu_control_threads];
  econ_read_finished = false;
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    econ_size[i] = 0;
    econ_buf[i].Allocate(kBufSize);
    pthread_mutex_init(&econ_buf_mutex[i], 0);
    pthread_cond_init(&econ_buf_full_cond[i], 0);
    pthread_cond_init(&econ_buf_empty_cond[i], 0);
  }

  // Init other members.
  in_edges_per_gpu = new unsigned int [num_gpu_control_threads];
  out_edges_per_gpu = new unsigned int [num_gpu_control_threads];
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    in_edges_per_gpu[i] = 0;
    out_edges_per_gpu[i] = 0;
  }

  // Since CUDA does not support non-empty constructors, we need to init
  // gpu_status manually.
  gpu_status.alive = true;
  gpu_status.have_message = false;
  pthread_mutex_init(&gpu_status_mutex, 0);
  pthread_cond_init(&gpu_status_condition, 0);

  // Initialize shared data for global shuffle.
  mcon_recv_ptrs = new MessageContent*[num_gpu_control_threads];
  mcon_recv_ptrs_mutex = new pthread_mutex_t[num_gpu_control_threads];
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    mcon_recv_ptrs[i] = NULL;
    pthread_mutex_init(&mcon_recv_ptrs_mutex[i], 0);
  }
  mcon_recv_used = new unsigned int[num_gpu_control_threads];

  // For performance test.
  pthread_mutex_init(&host_graph_mutex, 0);
}

SharedData::~SharedData() {
  cutDestroyBarrier(&global_barrier);
  cutDestroyBarrier(&vcon_memory_barrier);
  cutDestroyBarrier(&vcon_barrier);
  cutDestroyBarrier(&econ_memory_barrier);
  cutDestroyBarrier(&econ_barrier);
  cutDestroyBarrier(&all_control_threads_syn_barrier);
  cutDestroyBarrier(&superstep_end_barrier);

  const unsigned int num_gpu_control_threads = conf->GetNumGPUControlThreads();

  // Destory vcon buffer and related members.
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    vcon_size[i] = 0;
    vcon_buf[i].Deallocate();
    pthread_mutex_destroy(&vcon_buf_mutex[i]);
    pthread_cond_destroy(&vcon_buf_full_cond[i]);
    pthread_cond_destroy(&vcon_buf_empty_cond[i]);
  }
  delete[] vcon_size;
  delete[] vcon_buf;
  delete[] vcon_buf_mutex;
  delete[] vcon_buf_full_cond;
  delete[] vcon_buf_empty_cond;

  // Destory econ buffer and related members.
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    econ_size[i] = 0;
    econ_buf[i].Deallocate();
    pthread_mutex_destroy(&econ_buf_mutex[i]);
    pthread_cond_destroy(&econ_buf_full_cond[i]);
    pthread_cond_destroy(&econ_buf_empty_cond[i]);
  }
  delete[] econ_size;
  delete[] econ_buf;
  delete[] econ_buf_mutex;
  delete[] econ_buf_full_cond;
  delete[] econ_buf_empty_cond;

  delete[] in_edges_per_gpu;
  delete[] out_edges_per_gpu;

  pthread_mutex_destroy(&gpu_status_mutex);
  pthread_cond_destroy(&gpu_status_condition);

  // Release shared data for global shuffle.
  delete[] mcon_recv_ptrs;
  for (int i = 0; i < num_gpu_control_threads; ++i) {
    pthread_mutex_destroy(&mcon_recv_ptrs_mutex[i]);
  }
  delete[] mcon_recv_ptrs_mutex;
  delete[] mcon_recv_used;

  // Destroy performance test related data.
  pthread_mutex_destroy(&host_graph_mutex);
}

void SharedData::SignalGlobalRead() {
  cutIncrementBarrierAndWaitForBroadcast(&global_barrier, false);
}

void SharedData::WaitForGlobalRead() {
  cutWaitForBarrier(&global_barrier);
}

void SharedData::SignalVconMemoryReady() {
  cutIncrementBarrierAndWaitForBroadcast(&vcon_memory_barrier, false);
}

void SharedData::WaitForVconMemoryReady() {
  cutWaitForBarrier(&vcon_memory_barrier);
}

void SharedData::WaitForVconRead() {
  cutWaitForBarrier(&vcon_barrier);
}

void SharedData::SignalEconMemoryReady() {
  cutIncrementBarrierAndWaitForBroadcast(&econ_memory_barrier, false);
}

void SharedData::WaitForEconMemoryReady() {
  cutWaitForBarrier(&econ_memory_barrier);
}

void SharedData::WaitForEconRead() {
  cutWaitForBarrier(&econ_barrier);
}

void SharedData::WaitUntilAllGPUControlThreadsReachHere() {
  cutIncrementBarrierAndWaitForBroadcast(&all_control_threads_syn_barrier, true);
}

void SharedData::SignalSuperStepFinished() {
  cutIncrementBarrierAndWaitForBroadcast(&superstep_end_barrier, false);
}

void SharedData::WaitForSuperStepFinished() {
  cutWaitForBarrier(&superstep_end_barrier);
}

void SharedData::SetGlobalOnce(const IoGlobal &g) {
  pthread_mutex_lock(&host_graph_mutex);
  if (!global_set) {
    global = g;
    host_graph.SetGlobal(global);
    global_set = true;
  } else {
    cerr << "SharedData::SetGlobalOnce error: already set before!" << endl;
    exit(1);
  }
  pthread_mutex_unlock(&host_graph_mutex);
}

const IoGlobal& SharedData::GetGlobal() {
  return global;
}

void SharedData::AddVertex(const IoVertex &v) {
  // Lock corresponding buffer according to v.id.
  unsigned int gpu_id = kHashGetGPUId[conf->GetHashType()](
      global.num_vertex, conf->GetNumGPUControlThreads(), v.id);
  pthread_mutex_lock(&vcon_buf_mutex[gpu_id]);

  while (vcon_size[gpu_id] == kBufSize) {
    // If the buffer is full, send a 'full' signal.
    pthread_cond_signal(&vcon_buf_full_cond[gpu_id]);

    // Wait for the signal that indicates a copy finish.
    pthread_cond_wait(&vcon_buf_empty_cond[gpu_id], &vcon_buf_mutex[gpu_id]);
  }

  // Add v to that buffer
  vcon_buf[gpu_id].Set(vcon_size[gpu_id], v);
  ++vcon_size[gpu_id];
  out_edges_per_gpu[gpu_id] += v.out_edge_count;
  in_edges_per_gpu[gpu_id] += v.in_edge_count;

  // Unlock the buffer.
  pthread_mutex_unlock(&vcon_buf_mutex[gpu_id]);

  // Add to host graph for performance test.
  pthread_mutex_lock(&host_graph_mutex);
  host_graph.AddVertex(v);
  pthread_mutex_unlock(&host_graph_mutex);
}

void SharedData::FlushVconBuffer() {
  // If it is the last reading thread. Now there should be no more AddVertex
  // calls.
  if (cutIncrementBarrierAndWaitForBroadcast(&vcon_barrier, false)) {
    for (int gpu_id = 0; gpu_id < conf->GetNumGPUControlThreads(); ++gpu_id) {
      pthread_mutex_lock(&vcon_buf_mutex[gpu_id]);
    }

    vcon_read_finished = true;

    for (int gpu_id = 0; gpu_id < conf->GetNumGPUControlThreads(); ++gpu_id) {
      pthread_mutex_unlock(&vcon_buf_mutex[gpu_id]);
    }

    for (int gpu_id = 0; gpu_id < conf->GetNumGPUControlThreads(); ++gpu_id) {
      pthread_cond_signal(&vcon_buf_full_cond[gpu_id]);
    }
  }
}

bool SharedData::WaitForVconBufferReady(
    const unsigned int gpu_id,
    HostInVertexContent **buf,
    unsigned int *size) {
  pthread_mutex_lock(&vcon_buf_mutex[gpu_id]);
  // Not using while because only one single gpu control thread will operate on
  // a fixed gpu_id.
  if (vcon_size[gpu_id] < kBufSize && !vcon_read_finished) {
    pthread_cond_wait(&vcon_buf_full_cond[gpu_id], &vcon_buf_mutex[gpu_id]);
  }

  *buf = vcon_buf + gpu_id;
  *size = vcon_size[gpu_id];
  return vcon_read_finished;
}

void SharedData::SignalVconCopyFinish(const unsigned int gpu_id) {
  vcon_size[gpu_id] = 0;
  pthread_mutex_unlock(&vcon_buf_mutex[gpu_id]);
  pthread_cond_broadcast(&vcon_buf_empty_cond[gpu_id]);
}

void SharedData::AddEdge(const IoEdge &e) {
  unsigned int gpu_id = kHashGetGPUId[conf->GetHashType()](
      global.num_vertex, conf->GetNumGPUControlThreads(), e.from);
  pthread_mutex_lock(&econ_buf_mutex[gpu_id]);

  while (econ_size[gpu_id] == kBufSize) {
    pthread_cond_signal(&econ_buf_full_cond[gpu_id]);
    pthread_cond_wait(&econ_buf_empty_cond[gpu_id], &econ_buf_mutex[gpu_id]);
  }

  econ_buf[gpu_id].Set(econ_size[gpu_id], e);
  ++econ_size[gpu_id];
  pthread_mutex_unlock(&econ_buf_mutex[gpu_id]);

  // Add to host graph for performance test.
  pthread_mutex_lock(&host_graph_mutex);
  host_graph.AddEdge(e);
  pthread_mutex_unlock(&host_graph_mutex);
}

void SharedData::FlushEconBuffer() {
  if (cutIncrementBarrierAndWaitForBroadcast(&econ_barrier, false)) {
    for (int gpu_id = 0; gpu_id < conf->GetNumGPUControlThreads(); ++gpu_id) {
      pthread_mutex_lock(&econ_buf_mutex[gpu_id]);
    }

    econ_read_finished = true;

    for (int gpu_id = 0; gpu_id < conf->GetNumGPUControlThreads(); ++gpu_id) {
      pthread_mutex_unlock(&econ_buf_mutex[gpu_id]);
    }

    for (int gpu_id = 0; gpu_id < conf->GetNumGPUControlThreads(); ++gpu_id) {
      pthread_cond_signal(&econ_buf_full_cond[gpu_id]);
    }
  }
}

bool SharedData::WaitForEconBufferReady(
    const unsigned int gpu_id,
    HostInEdgeContent **buf,
    unsigned int *size) {
  pthread_mutex_lock(&econ_buf_mutex[gpu_id]);
  if (econ_size[gpu_id] < kBufSize && !econ_read_finished) {
    pthread_cond_wait(&econ_buf_full_cond[gpu_id], &econ_buf_mutex[gpu_id]);
  }

  *buf = econ_buf + gpu_id;
  *size = econ_size[gpu_id];
  return econ_read_finished;
}

void SharedData::SignalEconCopyFinish(const unsigned int gpu_id) {
  econ_size[gpu_id] = 0;
  pthread_mutex_unlock(&econ_buf_mutex[gpu_id]);
  pthread_cond_broadcast(&econ_buf_empty_cond[gpu_id]);
}

unsigned int SharedData::GetNumInEdgesForGPU(const unsigned int gpu_id) const {
  return in_edges_per_gpu[gpu_id];
}

unsigned int SharedData::GetNumOutEdgesForGPU(const unsigned int gpu_id) const {
  return out_edges_per_gpu[gpu_id];
}

void SharedData::GetGPUStatus(GPUStatus *status) const {
  status->CopyFrom(gpu_status);
}

void SharedData::MergeGPUStatusAndWait(const GPUStatus &status) {
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "SharedData::MergeGPUStatusAndWait..." << endl;
  cout << LAMBDA_HEADER << "barrier count: " << gpu_status_barrier_count << endl;
  );
#endif
  pthread_mutex_lock(&gpu_status_mutex);
  ++gpu_status_barrier_count;

  if (gpu_status_barrier_count == 1) gpu_status.Clear();
  gpu_status.MergeFrom(status);

  if (gpu_status_barrier_count == conf->GetNumGPUControlThreads()) {
#ifdef LAMBDA_DEBUG
    DBG_WRAP_COUT(
    cout << LAMBDA_HEADER << "MergeGPUStatusAndWait: status merged." << endl;
    );
#endif
    gpu_status_barrier_count = 0;
    pthread_mutex_unlock(&gpu_status_mutex);
    pthread_cond_broadcast(&gpu_status_condition);
  } else {
#ifdef LAMBDA_DEBUG
    DBG_WRAP_COUT(
    cout << LAMBDA_HEADER << "MergeGPUStatusAndWait: waiting.." << endl;
    );
#endif
    pthread_cond_wait(&gpu_status_condition, &gpu_status_mutex);
    pthread_mutex_unlock(&gpu_status_mutex);
  }
}

void SharedData::SetMsgRecvPtr(
    const unsigned int dest_gpu_id,
    MessageContent *mcon_recv_ptr) {
  mcon_recv_ptrs[dest_gpu_id] = mcon_recv_ptr;
}

unsigned int SharedData::GetMsgRecvCount(const unsigned int gpu_id) {
  return mcon_recv_used[gpu_id];
}

void SharedData::ResetMsgRecvCount(const unsigned int dest_gpu_id) {
  mcon_recv_used[dest_gpu_id] = 0;
}

unsigned int SharedData::GetMsgRecvPtr(
    const unsigned int dest_gpu_id,
    const unsigned int copy_size,
    MessageContent **mcon_recv_ptr) {
  pthread_mutex_lock(&mcon_recv_ptrs_mutex[dest_gpu_id]);

  unsigned int offset = mcon_recv_used[dest_gpu_id];
  mcon_recv_used[dest_gpu_id] += copy_size;
  (*mcon_recv_ptr) = mcon_recv_ptrs[dest_gpu_id];

  pthread_mutex_unlock(&mcon_recv_ptrs_mutex[dest_gpu_id]);
  return offset;
}

void SharedData::RunCPUSingleThreadAlgorithm(
    vector<HostGraphVertex> **host_graph_vertex,
    vector<HostGraphEdge> **host_graph_edge) {
  host_graph.RunAlgorithm();
  (*host_graph_vertex) = host_graph.GetVertexVector();
  (*host_graph_edge) = host_graph.GetEdgeVector();
}

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
