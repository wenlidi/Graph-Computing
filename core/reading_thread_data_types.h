// *****************************************************************************
// Filename:    reading_thread_data_types.h
// Date:        2012-12-07 19:16
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef READING_THREAD_DATA_TYPES_H_
#define READING_THREAD_DATA_TYPES_H_

#include <string>

#include "multithreading.h"
#include "gpu_control_thread_data_types.h"
#include "graph_reader.h"

using std::string;

class ReadingThreadData {
 public:

  ReadingThreadData();

  ~ReadingThreadData();

  void Init(
      const unsigned int id,
      const Config *conf,
      SharedData *shared_data);

  void Run();

 private:

  unsigned int reading_thread_id;
  const Config *conf;
  SharedData *shared_data;

  GraphReader *reader;

  // multi-threading control data, not owned by current class
  CUTBarrier *global_barrier;
  CUTBarrier *vcon_memory_barrier;
  CUTBarrier *vcon_barrier;
  CUTBarrier *econ_memory_barrier;
  CUTBarrier *econ_barrier;

  unsigned int num_gpu_control_threads;
  GPUControlThreadData *gpu_control_thread_data;  // not owned

};

#endif
