// *****************************************************************************
// Filename:    hash_functions.cc
// Date:        2012-12-11 14:09
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "host_hash_functions.h"

#include <algorithm>
#include <cmath>

#include "hash_types.h"
#include "util.h"

unsigned int MOD_GetGPUId(
    const unsigned int num_vertex,
    const unsigned int num_gpus,
    const unsigned int vid) {
  return vid % num_gpus;
}

unsigned int MOD_GetNumVertexForGPU(
    const unsigned int num_vertex,
    const unsigned int num_gpus,
    const unsigned int gpu_id) {
  return Util::GetCountForPartX(num_vertex, num_gpus, gpu_id);
}

unsigned int SPLIT_GetGPUId(
    const unsigned int num_vertex,
    const unsigned int num_gpus,
    const unsigned int vid) {
  unsigned int num_vertexes_per_gpu_upper =
      (num_vertex + num_gpus - 1) / num_gpus;
  unsigned int r = num_vertex % num_gpus;
  if (r == 0) {
    return vid / num_vertexes_per_gpu_upper;
  } else {
    return vid < r * num_vertexes_per_gpu_upper ?
        vid / num_vertexes_per_gpu_upper :
        r + (vid - r * num_vertexes_per_gpu_upper) / (num_vertexes_per_gpu_upper - 1);
  }
}

unsigned int SPLIT_GetNumVertexForGPU(
    const unsigned int num_vertex,
    const unsigned int num_gpus,
    const unsigned int gpu_id) {
  return Util::GetCountForPartX(num_vertex, num_gpus, gpu_id);
}
