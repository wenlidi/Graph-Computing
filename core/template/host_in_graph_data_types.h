// *****************************************************************************
// Filename:    host_in_graph_data_types.h
// Date:        2012-12-31 14:08
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: This file contains the data structures used by SharedData and
//              GraphReader to temporarily store read graph data from input and
//              copy to gpu.
// *****************************************************************************

#ifndef HOST_IN_GRAPH_DATA_TYPES_H_
#define HOST_IN_GRAPH_DATA_TYPES_H_

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "device_graph_data_types.h"

// Allocate write-combined memory on host.
#define ALLOCATE_IN_MEMBERS(TYPE, PTR, SIZE) \
    checkCudaErrors(cudaHostAlloc( \
            &PTR, \
            SIZE * sizeof(TYPE), \
            cudaHostAllocWriteCombined))

#define DEALLOCATE_ON_HOST(PTR) { \
    checkCudaErrors(cudaFreeHost(PTR)); \
    PTR = NULL; \
  }

// Need to be synchronized.
#define COPY_FROM_HOST_TO_DEVICE(MEMBER, DEVICE, DEVICE_OFFSET, COUNT, TYPE) \
    checkCudaErrors(cudaMemcpy( \
            DEVICE->d_##MEMBER + DEVICE_OFFSET, \
            MEMBER, \
            COUNT * sizeof(TYPE), \
            cudaMemcpyHostToDevice))

struct HostInVertexContent {
  unsigned int *id;
  unsigned int *in_edge_count;
  unsigned int *out_edge_count;
  //// TODO(laigd): add user defined 'in' members
$$V_IN[[<GP_TYPE> *<GP_NAME>;]]

  // Only allocate 'in' members
  void Allocate(const unsigned int size) {
    ALLOCATE_IN_MEMBERS(unsigned int, id, size);
    ALLOCATE_IN_MEMBERS(unsigned int, in_edge_count, size);
    ALLOCATE_IN_MEMBERS(unsigned int, out_edge_count, size);
    //// TODO(laigd): add user defined 'in' members
$$V_IN[[ALLOCATE_IN_MEMBERS(<GP_TYPE>, <GP_NAME>, size);]]
  }

  // Only deallocate 'in' members
  void Deallocate() {
    DEALLOCATE_ON_HOST(id);
    DEALLOCATE_ON_HOST(in_edge_count);
    DEALLOCATE_ON_HOST(out_edge_count);
    //// TODO(laigd): add user defined 'in' members
$$V_IN[[DEALLOCATE_ON_HOST(<GP_NAME>);]]
  }

  // Need to be synchronized.
  void CopyToDevice(
      const unsigned int device_offset,
      const unsigned int size,
      VertexContent *dest) {  // device memory
    COPY_FROM_HOST_TO_DEVICE(id,             dest, device_offset, size, unsigned int);
    COPY_FROM_HOST_TO_DEVICE(in_edge_count,  dest, device_offset, size, unsigned int);
    COPY_FROM_HOST_TO_DEVICE(out_edge_count, dest, device_offset, size, unsigned int);
    //// TODO(laigd): add user defined 'in' members
$$V_IN[[COPY_FROM_HOST_TO_DEVICE(<GP_NAME>, dest, device_offset, size, <GP_TYPE>);]]
  }

  // Used to get input.
  void Set(const unsigned int index, const IoVertex &v) {
    id[index] = v.id;
    in_edge_count[index] = v.in_edge_count;
    out_edge_count[index] = v.out_edge_count;
    //// TODO(laigd): add user defined 'in' members
$$V_IN[[<GP_NAME>[index] = v.<GP_NAME>;]]
  }
};

struct HostInEdgeContent {
  unsigned int *from;
  unsigned int *to;
  //// TODO(laigd): add user defined 'in' members
$$E_IN[[<GP_TYPE> *<GP_NAME>;]]

  // Only allocate 'in' members
  void Allocate(const unsigned int size) {
    ALLOCATE_IN_MEMBERS(unsigned int, from, size);
    ALLOCATE_IN_MEMBERS(unsigned int, to, size);
    //// TODO(laigd): add user defined 'in' members
$$E_IN[[ALLOCATE_IN_MEMBERS(<GP_TYPE>, <GP_NAME>, size);]]
  }

  // Only deallocate 'in' members
  void Deallocate() {
    DEALLOCATE_ON_HOST(from);
    DEALLOCATE_ON_HOST(to);
    //// TODO(laigd): add user defined 'in' members
$$E_IN[[DEALLOCATE_ON_HOST(<GP_NAME>);]]
  }

  void CopyToDevice(
      const unsigned int device_offset,
      const unsigned int size,
      EdgeContent *dest) {  // device memory
    COPY_FROM_HOST_TO_DEVICE(from, dest, device_offset, size, unsigned int);
    COPY_FROM_HOST_TO_DEVICE(to,   dest, device_offset, size, unsigned int);
    //// TODO(laigd): add user defined 'in' members
$$E_IN[[COPY_FROM_HOST_TO_DEVICE(<GP_NAME>, dest, device_offset, size, <GP_TYPE>);]]
  }

  void Set(const unsigned int index, const IoEdge &e) {
    from[index] = e.from;
    to[index] = e.to;
    //// TODO(laigd): add user defined 'in' members
$$E_IN[[<GP_NAME>[index] = e.<GP_NAME>;]]
  }
};

#undef ALLOCATE_IN_MEMBERS
#undef DEALLOCATE_ON_HOST
#undef COPY_FROM_HOST_TO_DEVICE

#endif
