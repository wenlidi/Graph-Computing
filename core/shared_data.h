// *****************************************************************************
// Filename:    shared_data.h
// Date:        2012-12-12 14:00
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef SHARED_DATA_H_
#define SHARED_DATA_H_

#include "multithreading.h"
#include "generated_io_data_types.h"
#include "config.h"
#include "gpu_status.h"
#include "host_graph_data_types.h"
#include "host_in_graph_data_types.h"
#include "host_graph.h"

// Multithreading shared data type.
class SharedData {
 public:

  SharedData(const Config *config);

  ~SharedData();

  void SignalGlobalRead();
  void WaitForGlobalRead();

  void SignalVconMemoryReady();
  void WaitForVconMemoryReady();

  void WaitForVconRead();

  void SignalEconMemoryReady();
  void WaitForEconMemoryReady();

  void WaitForEconRead();

  void WaitUntilAllGPUControlThreadsReachHere();

  void SignalSuperStepFinished();
  void WaitForSuperStepFinished();

  void SetGlobalOnce(const IoGlobal &g);
  const IoGlobal& GetGlobal();

  // Graph reader use this function to add a vertex after read it.
  void AddVertex(const IoVertex &v);

  // Graph reader use this function to tell gpu control threads that all data
  // have been read to buffer.
  void FlushVconBuffer();

  // The gpu control threads use this function to wait for a vcon buffer.
  // Returns true if this should be the last call (i.e. all vcon are read and
  // there will be no more Vertex added to buffer).
  bool WaitForVconBufferReady(
      const unsigned int gpu_id,
      HostInVertexContent **buf,
      unsigned int *size);

  // The gpu control threads use this function to signal that the previous
  // waited buffer has been copied to device memory.
  void SignalVconCopyFinish(const unsigned int gpu_id);

  // The following 4 functions are similar to the ones for Vertex as declared
  // above.
  void AddEdge(const IoEdge &e);
  void FlushEconBuffer();
  bool WaitForEconBufferReady(
      const unsigned int gpu_id,
      HostInEdgeContent **buf,
      unsigned int *size);
  void SignalEconCopyFinish(const unsigned int gpu_id);

  unsigned int GetNumInEdgesForGPU(const unsigned int gpu_id) const;

  unsigned int GetNumOutEdgesForGPU(const unsigned int gpu_id) const;

  void GetGPUStatus(GPUStatus *status) const;

  void MergeGPUStatusAndWait(const GPUStatus &status);

  // Functions for global shuffle.
  void SetMsgRecvPtr(
      const unsigned int dest_gpu_id,
      MessageContent *mcon_recv_ptr);

  unsigned int GetMsgRecvCount(const unsigned int gpu_id);

  void ResetMsgRecvCount(const unsigned int dest_gpu_id);

  // Returns the offset.
  unsigned int GetMsgRecvPtr(
      const unsigned int dest_gpu_id,
      const unsigned int copy_size,
      MessageContent **mcon_recv_ptr);

  // Single-threaded access.
  void RunCPUSingleThreadAlgorithm(
      vector<HostGraphVertex> **host_graph_vertex,
      vector<HostGraphEdge> **host_graph_edge);

 private:

  static const unsigned int kBufSize = 256;

  const Config *conf;

  // Shared between gpu control threads, reading threads and main thread.
  CUTBarrier global_barrier;
  CUTBarrier vcon_memory_barrier;
  CUTBarrier vcon_barrier;
  CUTBarrier econ_memory_barrier;
  CUTBarrier econ_barrier;

  // Shared between gpu control threads and main thread.
  CUTBarrier all_control_threads_syn_barrier;
  CUTBarrier superstep_end_barrier;
  CUTBarrier gpu_status_merge_barrier;

  // IoGlobal
  IoGlobal global;
  bool global_set;

  // About read/copy vcon buffer.
  unsigned int *vcon_size;
  HostInVertexContent *vcon_buf;
  pthread_mutex_t *vcon_buf_mutex;
  pthread_cond_t *vcon_buf_full_cond;
  pthread_cond_t *vcon_buf_empty_cond;
  bool vcon_read_finished;

  // About read/copy econ buffer.
  unsigned int *econ_size;
  HostInEdgeContent *econ_buf;
  pthread_mutex_t *econ_buf_mutex;
  pthread_cond_t *econ_buf_full_cond;
  pthread_cond_t *econ_buf_empty_cond;
  bool econ_read_finished;

  // When reading vertex data, we sum up the number of out edges for each gpu.
  unsigned int *in_edges_per_gpu;
  unsigned int *out_edges_per_gpu;

  // GPUStatus
  GPUStatus gpu_status;
  unsigned int gpu_status_barrier_count;
  pthread_mutex_t gpu_status_mutex;
  pthread_cond_t gpu_status_condition;

  // Shared data for global shuffle.
  MessageContent **mcon_recv_ptrs;
  pthread_mutex_t *mcon_recv_ptrs_mutex;
  unsigned int *mcon_recv_used;  // Need to be reset for each super step.

  // For performance test vs. cpu algorithm.
  pthread_mutex_t host_graph_mutex;
  HostGraph host_graph;
};

#endif
