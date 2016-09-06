// *****************************************************************************
// Filename:    device_hash_functions.h
// Date:        2012-12-22 23:31
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef DEVICE_HASH_FUNCTIONS_H_
#define DEVICE_HASH_FUNCTIONS_H_

// NOTE: This file could only be involved in one and only one .cu file.

__device__ unsigned int GetNumVertexForGPUX(unsigned int gpu_x_id) {
  return d_global.d_num_vertex / d_auxiliary.d_num_gpus
      + (gpu_x_id < d_global.d_num_vertex % d_auxiliary.d_num_gpus ? 1 : 0);
}

__device__ unsigned int D_MOD_GetGPUId(unsigned int vid) {
  return vid % d_auxiliary.d_num_gpus;
}

__device__ unsigned int D_MOD_GetNumVertexForGPU(
    unsigned int gpu_id) {
  return GetNumVertexForGPUX(gpu_id);
}

__device__ unsigned int D_SPLIT_GetGPUId(unsigned int vid) {
  unsigned int num_vertexes_per_gpu_upper =
      (d_global.d_num_vertex + d_auxiliary.d_num_gpus - 1) /
      d_auxiliary.d_num_gpus;
  unsigned int r = d_global.d_num_vertex % d_auxiliary.d_num_gpus;
  if (r == 0) {
    return vid / num_vertexes_per_gpu_upper;
  } else {
    return vid < r * num_vertexes_per_gpu_upper ?
        vid / num_vertexes_per_gpu_upper :
        r + (vid - r * num_vertexes_per_gpu_upper) /
            (num_vertexes_per_gpu_upper - 1);
  }
}

__device__ unsigned int D_SPLIT_GetNumVertexForGPU(
    unsigned int gpu_id) {
  return GetNumVertexForGPUX(gpu_id);
}

#endif
