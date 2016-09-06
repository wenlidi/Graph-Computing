// *****************************************************************************
// Filename:    auxiliary_manager.cc
// Date:        2013-01-08 09:36
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "auxiliary_manager.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "constants.h"
#include "device_graph_data_types.h"
#include "device_util.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> "
#endif

void AuxiliaryManager::Allocate(
    const unsigned int num_gpus,
    const unsigned int vcon_array_size,
    const unsigned int econ_array_size,
    const unsigned int mcon_recv_array_size,
    AuxiliaryDeviceData *auxiliary) {
  auxiliary->d_num_gpus = num_gpus;

  ALLOCATE_ON_DEVICE(unsigned int, auxiliary->d_in_msg_from,         mcon_recv_array_size);
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
  ALLOCATE_ON_DEVICE(unsigned int, auxiliary->d_in_msg_next,         mcon_recv_array_size);
#endif
  ALLOCATE_ON_DEVICE(unsigned int, auxiliary->d_out_edge_in_msg_map, econ_array_size);

  unsigned int max_size =
      std::max(vcon_array_size,
               std::max(econ_array_size,
                        mcon_recv_array_size));
}

void AuxiliaryManager::Deallocate(AuxiliaryDeviceData *auxiliary) {
  DEALLOCATE_ON_DEVICE(auxiliary->d_in_msg_from);
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
  DEALLOCATE_ON_DEVICE(auxiliary->d_in_msg_next);
#endif
  DEALLOCATE_ON_DEVICE(auxiliary->d_out_edge_in_msg_map);
}

#ifdef LAMBDA_DEBUG
void AuxiliaryManager::DebugOutput(
    const AuxiliaryDeviceData &auxiliary,
    const unsigned int vcon_array_size,
    const unsigned int econ_array_size,
    const unsigned int mcon_recv_array_size) {
  unsigned int *buf = NULL;
  unsigned int max_size =
      std::max(vcon_array_size,
               std::max(econ_array_size,
                        mcon_recv_array_size));
  checkCudaErrors(cudaMallocHost(&buf, max_size * sizeof(unsigned int)));

  cout << LAMBDA_HEADER << "[AuxiliaryDeviceData]" << endl;
  DEBUG_OUTPUT(buf, auxiliary.d_in_msg_from,         "in_msg_from:         ", mcon_recv_array_size, unsigned int);
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
  DEBUG_OUTPUT(buf, auxiliary.d_in_msg_next,         "in_msg_next:         ", mcon_recv_array_size, unsigned int);
#endif
  DEBUG_OUTPUT(buf, auxiliary.d_out_edge_in_msg_map, "out_edge_in_msg_map: ", econ_array_size,      unsigned int);

  checkCudaErrors(cudaFreeHost(buf));
}
#endif

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
