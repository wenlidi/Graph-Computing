// *****************************************************************************
// Filename:    gpu_control_thread.cc
// Date:        2012-12-07 15:08
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "gpu_control_thread.h"

#include "gpu_control_thread_data_types.h"

void* GPUControlThread(void *args) {
  GPUControlThreadData *data = (GPUControlThreadData*)args;
  data->Run();
  return 0;
}
