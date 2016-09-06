// *****************************************************************************
// Filename:    host_out_graph_data_types.h
// Date:        2012-12-31 14:04
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: This file contains data structures used by
//              SingleStreamWriter to accumulate the computation result and
//              output to an ostream.
// *****************************************************************************

#ifndef HOST_OUT_GRAPH_DATA_TYPES_H_
#define HOST_OUT_GRAPH_DATA_TYPES_H_

#include <vector>
#include <ostream>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "device_graph_data_types.h"
#include "host_graph_data_types.h"

using std::ostream;
using std::vector;
using std::cout;
using std::endl;

#define ALLOCATE_OUT_MEMBERS(TYPE, PTR, SIZE) \
    checkCudaErrors(cudaMallocHost(&PTR, SIZE * sizeof(TYPE)))

#define DEALLOCATE_ON_HOST(PTR) { \
    checkCudaErrors(cudaFreeHost(PTR)); \
    PTR = NULL; \
  }

#define COPY_FROM_DEVICE_TO_HOST(MEMBER, HOST_OFFSET, DEVICE, DEVICE_OFFSET, COUNT, TYPE) \
    checkCudaErrors(cudaMemcpy( \
            MEMBER + HOST_OFFSET, \
            DEVICE.d_##MEMBER + DEVICE_OFFSET, \
            COUNT * sizeof(TYPE), \
            cudaMemcpyDeviceToHost))

class HostOutGlobal {
 public:
  static void Write(const Global &g, ostream &out) {
    out << "num_vertex: " << g.d_num_vertex << ", "
        << "num_edge: " << g.d_num_edge
        //// TODO(laigd): add user defined members
$$G[[<< ", " << "<GP_NAME>: " << g.d_<GP_NAME>]]
        << endl;
  }
};

class HostOutVertexContent {
 public:

  // Members we need to copy.
  unsigned int *id;
  //// TODO(laigd): add user defined 'out' members
$$V_OUT[[<GP_TYPE> *<GP_NAME>;]]

  HostOutVertexContent()
      : capacity(0),
        size(0),
        // Members we need to copy from device.
        id(NULL)
        //// TODO(laigd): add user defined 'out' members
$$V_OUT[[, <GP_NAME>(NULL)]]
  {
  }

  ~HostOutVertexContent() {
    if (id != NULL) Deallocate();
  }

  // Only allocate 'out' members
  void Allocate(const unsigned int cap) {
    if (id != NULL) {
      cout << "HostOutVertexContent error: already allocated!" << endl;
      exit(1);
    }

    capacity = cap;
    size = 0;

    ALLOCATE_OUT_MEMBERS(unsigned int, id, capacity);
    //// TODO(laigd): add user defined 'out' members
$$V_OUT[[ALLOCATE_OUT_MEMBERS(<GP_TYPE>, <GP_NAME>, capacity);]]
  }

  // Only deallocate 'out' members
  void Deallocate() {
    if (id == NULL) {
      cout << "HostOutVertexContent error: deallocate empty object!" << endl;
      exit(1);
    }

    capacity = 0;
    size = 0;

    DEALLOCATE_ON_HOST(id);
    //// TODO(laigd): add user defined 'out' members
$$V_OUT[[DEALLOCATE_ON_HOST(<GP_NAME>);]]
  }

  void CopyFromDevice(
      const VertexContent &src,
      const unsigned int device_offset,
      const unsigned int copy_size) {
    if (size + copy_size > capacity) {
      cout << "HostOutVertexContent error: copy too much content!"
           << "  Capacity: " << capacity
           << ", Current size: " << size
           << ", Copy size: " << copy_size
           << endl;
      exit(1);
    }

    COPY_FROM_DEVICE_TO_HOST(id, size, src, device_offset, copy_size, unsigned int);
    //// TODO(laigd): add user defined 'out' members
$$V_OUT[[COPY_FROM_DEVICE_TO_HOST(<GP_NAME>, size, src, device_offset, copy_size, <GP_TYPE>);]]

    size += copy_size;
  }

  void Clear() {
    size = 0;
  }

  unsigned int Size() const {
    return size;
  }

  unsigned int Capacity() const {
    return capacity;
  }

  // Sort the content according to id.
  void SortById() {
    int *sort_map;
    ALLOCATE_OUT_MEMBERS(int, sort_map, capacity);
    for (unsigned int i = 0; i < size; ++i) {
      sort_map[i] = i;
    }
    std::sort(sort_map, sort_map + size, HostOutVertexContentCmp(id));

    for (unsigned int i = 0; i < size; ++i) {
      if (sort_map[i] != -1) {
        unsigned int tmp_id = id[i];
        //// TODO(laigd): Add user defined 'out' members.
$$V_OUT[[<GP_TYPE> tmp_<GP_NAME> = <GP_NAME>[i];]]
        unsigned int q = i, p = sort_map[i];
        while (p != i) {
          id[q] = id[p];
          //// TODO(laigd): Add user defined 'out' members.
$$V_OUT[[<GP_NAME>[q] = <GP_NAME>[p];]]
          sort_map[q] = -1;
          q = p;
          p = sort_map[p];
        }
        id[q] = tmp_id;
        //// TODO(laigd): Add user defined 'out' members.
$$V_OUT[[<GP_NAME>[q] = tmp_<GP_NAME>;]]
        sort_map[q] = -1;
      }
    }
    DEALLOCATE_ON_HOST(sort_map);
  }

  void Write(ostream &out) {
    for (unsigned int i = 0; i < size; ++i) {
      out << id[i]
          //// TODO(laigd): Add user defined 'out' members.
$$V_OUT[[<< ", " << <GP_NAME>[i]]]
          << endl;
    }
  }

 private:

  unsigned int capacity;
  unsigned int size;

  struct HostOutVertexContentCmp {
    const unsigned int *id;

    HostOutVertexContentCmp(const unsigned int *id_ptr) : id(id_ptr) {
    }

    // HostOutVertexContentCmp(const HostOutVertexContentCmp &other)
    //     : id(other.id) {
    // }

    bool operator()(const unsigned int i, const unsigned int j) {
      return id[i] < id[j];
    }
  };

};

struct HostOutEdgeContent {
 public:

  // Members we need to copy.
  unsigned int *from;
  unsigned int *to;
  //// TODO(laigd): add user defined 'out' members
$$E_OUT[[<GP_TYPE> *<GP_NAME>;]]

  HostOutEdgeContent()
      : capacity(0),
        size(0),
        // Members we need to copy from device.
        from(NULL),
        to(NULL)
        //// TODO(laigd): add user defined 'out' members
$$E_OUT[[, <GP_NAME>(NULL)]]
  {
  }

  ~HostOutEdgeContent() {
    if (from != NULL) Deallocate();
  }

  // Only allocate 'out' members
  void Allocate(const unsigned int cap) {
    if (from != NULL) {
      cout << "HostOutEdgeContent error: already allocated!" << endl;
      exit(1);
    }

    capacity = cap;
    size = 0;

    ALLOCATE_OUT_MEMBERS(unsigned int, from, capacity);
    ALLOCATE_OUT_MEMBERS(unsigned int, to, capacity);
    //// TODO(laigd): add user defined 'out' members
$$E_OUT[[ALLOCATE_OUT_MEMBERS(<GP_TYPE>, <GP_NAME>, capacity);]]
  }

  // Only deallocate 'out' members
  void Deallocate() {
    if (from == NULL) {
      cout << "HostOutEdgeContent error: deallocate empty object!" << endl;
      exit(1);
    }

    capacity = 0;
    size = 0;

    DEALLOCATE_ON_HOST(from);
    DEALLOCATE_ON_HOST(to);
    //// TODO(laigd): add user defined 'out' members
$$E_OUT[[DEALLOCATE_ON_HOST(<GP_NAME>);]]
  }

  void CopyFromDevice(
      const EdgeContent &src,
      const unsigned int device_offset,
      const unsigned int copy_size) {
    if (size + copy_size > capacity) {
      cout << "HostOutEdgeContent error: copy too much content!"
           << "  Capacity: " << capacity
           << ", Current size: " << size
           << ", Copy size: " << copy_size
           << endl;
      exit(1);
    }

    COPY_FROM_DEVICE_TO_HOST(from, size, src, device_offset, copy_size, unsigned int);
    COPY_FROM_DEVICE_TO_HOST(to,   size, src, device_offset, copy_size, unsigned int);
    //// TODO(laigd): add user defined 'out' members
$$E_OUT[[COPY_FROM_DEVICE_TO_HOST(<GP_NAME>, size, src, device_offset, copy_size, <GP_TYPE>);]]

    size += copy_size;
  }

  void Clear() {
    size = 0;
  }

  unsigned int Size() const {
    return size;
  }

  unsigned int Capacity() const {
    return capacity;
  }

  // Sort the content according to from and to.
  void SortByFromTo() {
    int *sort_map;
    ALLOCATE_OUT_MEMBERS(unsigned int, sort_map, capacity);
    for (unsigned int i = 0; i < size; ++i) {
      sort_map[i] = i;
    }
    std::sort(sort_map, sort_map + size, HostOutEdgeContentCmp(from, to));

    for (unsigned int i = 0; i < size; ++i) {
      if (sort_map[i] != -1) {
        unsigned int tmp_from = from[i];
        unsigned int tmp_to = to[i];
        //// TODO(laigd): Add user defined 'out' members.
$$E_OUT[[<GP_TYPE> tmp_<GP_NAME> = <GP_NAME>[i];]]
        unsigned int q = i, p = sort_map[i];
        while (p != i) {
          from[q] = from[p];
          to[q] = to[p];
          //// TODO(laigd): Add user defined 'out' members.
$$E_OUT[[<GP_NAME>[q] = <GP_NAME>[p];]]
          sort_map[q] = -1;
          q = p;
          p = sort_map[p];
        }
        from[q] = tmp_from;
        to[q] = tmp_to;
        //// TODO(laigd): Add user defined 'out' members.
$$E_OUT[[<GP_NAME>[q] = tmp_<GP_NAME>;]]
        sort_map[q] = -1;
      }
    }
    DEALLOCATE_ON_HOST(sort_map);
  }

  void Write(ostream &out) {
    for (unsigned int i = 0; i < size; ++i) {
      out << from[i] << ", " << to[i]
          //// TODO(laigd): Add user defined 'out' members.
$$E_OUT[[<< ", " << <GP_NAME>[i]]]
          << endl;
    }
  }

 private:

  unsigned int capacity;
  unsigned int size;

  struct HostOutEdgeContentCmp {
    const unsigned int *from, *to;

    HostOutEdgeContentCmp(
        const unsigned int *from_ptr,
        const unsigned int *to_ptr)
        : from(from_ptr), to(to_ptr) {
    }

    bool operator()(const unsigned int i, const unsigned int j) {
      return (from[i] == from[j] ? to[i] < to[j] : from[i] < from[j]);
    }
  };

};

#undef ALLOCATE_OUT_MEMBERS
#undef DEALLOCATE_ON_HOST
#undef COPY_FROM_DEVICE_TO_HOST

#endif
