// *****************************************************************************
// Filename:    init.h
// Date:        2013-03-01 19:53
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef INIT_H_
#define INIT_H_

#include <gflags/gflags.h>

DECLARE_int32(num_gpus);
DECLARE_int32(single_gpu_id);
DECLARE_int32(max_superstep);
DECLARE_int32(num_threads_per_block);
DECLARE_string(input_file);
DECLARE_string(graph_type);
DECLARE_string(hash_type);
DECLARE_string(output_file);
DECLARE_string(writer_type);

DECLARE_int32(rand_num_vertex);
DECLARE_int32(rand_num_edge);
DECLARE_int32(rand_num_reading_threads);

class Config;

int GetGPUInfo(int *available_device_id);

void SetConfigByCmdFlags(Config *conf);

void RunGPregel(
    const Config &conf,
    const int *available_device_id,
    const int num_available_device);

#endif
