// *****************************************************************************
// Filename:    gpu_control_thread_data_types.h
// Date:        2012-12-06 15:37
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GPU_CONTROL_THREAD_DATA_TYPE_H_
#define GPU_CONTROL_THREAD_DATA_TYPE_H_

#include "config.h"
#include "device_graph_data_types.h"
#include "gpu_status.h"
#include "gpu_storage.h"
#include "output_writer.h"
#include "shared_data.h"

class GPUControlThreadData {
 public:

  GPUControlThreadData();

  ~GPUControlThreadData();

  void Init(
      const int id,
      const int device_id,
      const Config *conf,
      SharedData *shared_data);

  void Run();

 private:

  int control_thread_id;

  // CUDA device managememt data.
  int cuda_device_id;

  // Controling data
  const Config *conf;
  SharedData *shared_data;

  // Host view of device data
  Global global;
  VertexContent vcon;
  EdgeContent econ;
  MessageContent mcon_recv;
  MessageContent mcon_send;
  AuxiliaryDeviceData auxiliary;

  unsigned int vcon_copied_size;
  unsigned int econ_copied_size;

  GPUStatus local_gpu_status;

  GPUStorageManager gpu_storage_manager;

  OutputWriter *writer;

  void WaitAndCopyVconToGPU();
  void WaitAndCopyEconToGPU();

  void PrepareSuperStep();

  void RunSuperStepsOnSingleGPU();

  void RunSuperStepsOnMultipleGPUs();

  void MultiGPUGlobalShuffle(
      const unsigned int *msg_per_gpu_begin,
      const unsigned int *msg_per_gpu_end);
};

#endif
