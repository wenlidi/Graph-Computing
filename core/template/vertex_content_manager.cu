// *****************************************************************************
// Filename:    vertex_content_manager.cc
// Date:        2013-01-08 10:10
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "vertex_content_manager.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>

#include "constants.h"
#include "device_graph_data_types.h"
#include "device_util.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> "
#endif

void VertexContentManager::Allocate(
    const unsigned int size,
    VertexContent *vcon) {
  vcon->d_size = size;
  ALLOCATE_ON_DEVICE(unsigned int, vcon->d_id,             vcon->d_size);
  ALLOCATE_ON_DEVICE(unsigned int, vcon->d_in_edge_count,  vcon->d_size);
  ALLOCATE_ON_DEVICE(unsigned int, vcon->d_out_edge_count, vcon->d_size);
  //// TODO(laigd): add user defined members
$$V[[ALLOCATE_ON_DEVICE(<GP_TYPE>, vcon->d_<GP_NAME>, vcon->d_size);]]
}

void VertexContentManager::Deallocate(VertexContent *vcon) {
  DEALLOCATE_ON_DEVICE(vcon->d_id);
  DEALLOCATE_ON_DEVICE(vcon->d_in_edge_count);
  DEALLOCATE_ON_DEVICE(vcon->d_out_edge_count);
  //// TODO(laigd): add user defined members
$$V[[DEALLOCATE_ON_DEVICE(vcon->d_<GP_NAME>);]]
}

void VertexContentManager::ShuffleInMembers(
    VertexContent *vcon,
    thrust::device_ptr<unsigned int> &thr_shuffle_index,
    void *d_tmp_buf) {
  SHUFFLE_MEMBER(unsigned int, vcon->d_id,             vcon->d_size, d_tmp_buf, thr_shuffle_index);
  SHUFFLE_MEMBER(unsigned int, vcon->d_in_edge_count,  vcon->d_size, d_tmp_buf, thr_shuffle_index);
  SHUFFLE_MEMBER(unsigned int, vcon->d_out_edge_count, vcon->d_size, d_tmp_buf, thr_shuffle_index);
  //// TODO(laigd): add user defined 'in' members
$$V_IN[[SHUFFLE_MEMBER(<GP_TYPE>, vcon->d_<GP_NAME>, vcon->d_size, d_tmp_buf, thr_shuffle_index);]]
}

void VertexContentManager::InitOutMembers(VertexContent *vcon) {
  //// TODO(laigd): Add user defined 'out' members
$$V_OUT[[INIT_OUT_MEMBERS(<GP_TYPE>, vcon->d_<GP_NAME>, vcon->d_size, <GP_INIT_VALUE>);]]
}

#ifdef LAMBDA_DEBUG
void VertexContentManager::DebugOutput(const VertexContent &vcon) {
  unsigned int *buf = NULL;
  checkCudaErrors(cudaMallocHost(&buf, vcon.d_size * sizeof(unsigned int)));

  cout << LAMBDA_HEADER << "[VertexContent]" << endl;
  DEBUG_OUTPUT(buf, vcon.d_id,             "id:             ", vcon.d_size, unsigned int);
  DEBUG_OUTPUT(buf, vcon.d_in_edge_count,  "in_edge_count:  ", vcon.d_size, unsigned int);
  DEBUG_OUTPUT(buf, vcon.d_out_edge_count, "out_edge_count: ", vcon.d_size, unsigned int);
  //// TODO(laigd): add user defined members
$$V[[DEBUG_OUTPUT(buf, vcon.d_<GP_NAME>, "<GP_NAME>: ", vcon.d_size, <GP_TYPE>);]]

  checkCudaErrors(cudaFreeHost(buf));
}
#endif

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
