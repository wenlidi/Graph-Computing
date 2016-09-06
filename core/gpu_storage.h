// *****************************************************************************
// Filename:    gpu_storage.h
// Date:        2012-12-06 16:11
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GPU_STORAGE_H_
#define GPU_STORAGE_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "device_graph_data_types.h"
#include "gpu_status.h"
#include "config.h"
#include "constants.h"

class GPUStorageManager {
 public:

  GPUStorageManager();

  ~GPUStorageManager();

  void Init(const Config *c);

  // Below are general data transfer functions between host and device.

  void GetGPUStatusFromGPU(GPUStatus *gpu_status);

  void CopyGPUStatusToGPU(const GPUStatus &gpu_status);

  void CopyGlobalToGPU(const Global &global, cudaStream_t stream = 0);

  void CopyVconToGPU(const VertexContent &vcon, cudaStream_t stream = 0);

  void CopyEconToGPU(const EdgeContent &econ, cudaStream_t stream = 0);

  void CopyMconSendToGPU(const MessageContent &mcon_send, cudaStream_t stream = 0);

  void CopyMconRecvToGPU(const MessageContent &mcon_recv, cudaStream_t stream = 0);

  void CopyAuxiliaryToGPU(const AuxiliaryDeviceData &auxiliary, cudaStream_t stream = 0);

  // Below are data management functions.

  void SingleGPUBuildIndexes(
      VertexContent *vcon,
      EdgeContent *econ,
      AuxiliaryDeviceData *auxiliary);

  void UserCompute(VertexContent *vcon);

 private:

  // TODO(laigd): This value should be determine by the type of GPU dynamically.
  static const unsigned int kDefaultNumThreadsPerBlock = 256;

  const Config *conf;

  bool msg_per_gpu_begin_calculated;
  unsigned int *msg_per_gpu_begin;
  unsigned int *msg_per_gpu_end;

  unsigned int GetNumBlocks(const unsigned int num_threads_need) {
    return (num_threads_need + kDefaultNumThreadsPerBlock - 1)
        / kDefaultNumThreadsPerBlock;
  }

  void SingleGPUBuildVconIndexes(VertexContent *vcon);

  void SingleGPUBuildEconIndexes(
      const unsigned int *d_vid_index,
      const unsigned int num_vertexes,
      EdgeContent *econ);

#ifdef LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT
  void SingleGPUBuildAuxiliaryIndexes(
      const unsigned int *d_vid_index,
      const VertexContent &vcon,
      const EdgeContent &econ,
      AuxiliaryDeviceData *auxiliary);
#else
  void SingleGPUBuildAuxiliaryIndexes(
      const VertexContent &vcon,
      const EdgeContent &econ,
      AuxiliaryDeviceData *auxiliary);
#endif
};

#endif
