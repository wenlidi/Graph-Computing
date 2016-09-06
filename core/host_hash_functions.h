// *****************************************************************************
// Filename:    hash_functions.h
// Date:        2012-12-11 14:08
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef HASH_FUNCTIONS_H_
#define HASH_FUNCTIONS_H_

typedef unsigned int uint;
typedef const unsigned int kuint;
typedef uint (*HashFuncType)(kuint n1, kuint n2, kuint n3);

uint MOD_GetGPUId(kuint num_vertex, kuint num_gpus, kuint vid);
uint MOD_GetNumVertexForGPU(kuint num_vertex, kuint num_gpus, kuint gpu_id);
uint SPLIT_GetGPUId(kuint num_vertex, kuint num_gpus, kuint vid);
uint SPLIT_GetNumVertexForGPU(kuint num_vertex, kuint num_gpus, kuint gpu_id);

const HashFuncType kHashGetGPUId[2] = {
  MOD_GetGPUId,
  SPLIT_GetGPUId,
};

const HashFuncType kHashGetNumVertexForGPU[2] = {
  MOD_GetNumVertexForGPU,
  SPLIT_GetNumVertexForGPU,
};

#endif
