// *****************************************************************************
// Filename:    global_manager.cc
// Date:        2013-01-08 10:01
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "global_manager.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "device_graph_data_types.h"
#include "generated_io_data_types.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> "
#endif

void GlobalManager::Set(const IoGlobal &src, Global *dst) {
  dst->d_num_vertex = src.num_vertex;
  dst->d_num_edge = src.num_edge;
  //// TODO(laigd): add user defined members
$$G[[dst->d_<GP_NAME> = src.<GP_NAME>;]]
}

#ifdef LAMBDA_DEBUG
void GlobalManager::DebugOutput(const Global &global) {
  cout << LAMBDA_HEADER << "[Global]" << endl;
  cout << LAMBDA_HEADER
      << "num_vertex: " << global.d_num_vertex << ", "
      << "num_edge: " << global.d_num_edge
      //// TODO(laigd): add user defined members
$$G[[<< ", " << "<GP_NAME>: " << global.d_<GP_NAME>]]
      << endl;
}
#endif

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
