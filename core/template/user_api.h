// *****************************************************************************
// Filename:    user_api.h
// Date:        2012-12-25 13:52
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef USER_API_H_
#define USER_API_H_

// NOTE: This file could only be involved in one and only one .cu file.

#include "device_constants.h"

struct Edge {
  // Pointing to each member of econ
  unsigned int index;

  /*************************** system members' get/set ************************/
  // 'from' should not be discarded because some users may want to defined
  // kernels to process all edges.
  __device__ unsigned int get_from() const {
    return d_econ.d_from[index];
  }

  __device__ unsigned int get_to() const {
    return d_econ.d_to[index];
  }

  /************************ user defined members' get/set *********************/
  //// TODO(laigd): add user defined members' GET/SET
$$E[[__device__ <GP_TYPE> get_<GP_NAME>() const { return d_econ.d_<GP_NAME>[index]; }]]
$$E_OUT[[__device__ void set_<GP_NAME>(const <GP_TYPE> &value) { d_econ.d_<GP_NAME>[index] = value; }]]
};

// Out message.
struct Message {
  // Pointing to each member of mcon_send
  unsigned int index;

  // constructor, set from and to according to the attached edge (and set all
  // user defined members to invalid state if required)
  __device__ Message(const Edge &e) {
    index = d_auxiliary.d_out_edge_in_msg_map[e.index];
  }

  /***************************** api functions ********************************/
  // Set have_message to 1
  __device__ void Send() {
#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    d_mcon_send.d_is_full[index] = true;
#endif
    d_gpu_status.have_message = true;
  }

  /************************ user defined members' set *************************/
  //// TODO(laigd): add user defined members' SET
$$M[[__device__ void set_<GP_NAME>(const <GP_TYPE> &value) { d_mcon_send.d_<GP_NAME>[index] = value; }]]
};

struct OutEdgeIterator {
  unsigned int vertex_index;
  Edge e;

  __device__ OutEdgeIterator(const unsigned int vidx) {
    vertex_index = vidx;
    e.index = (vidx == 0 ? 0 : d_vcon.d_out_edge_count[vidx - 1]);
  }

  /***************************** api functions ********************************/
  // since edges with same 'from' are arranged adjacently, so we only need to
  // see whether e.index + 1 >= out_edge_count[vid].
  __device__ bool Done() {
    return e.index == d_vcon.d_out_edge_count[vertex_index];
  }

  // Simply let e.index++
  // TODO(laigd): Maybe we can use the edge storage scheme described in the
  // Medusa paper to enable coalesed memory accessing.
  __device__ void Next() {
    ++e.index;
  }
  __device__ void AddOffset(int offset){
	e.index+=offset;
  }

  // returns &e.
  __device__ Edge* operator->() {
    return &e;
  }

  // returns e.
  __device__ Edge& operator*() {
    return e;
  }
};

// In message
struct MessageIterator {
  // Pointing to each member of mcon_recv.
  unsigned int mcon_recv_idx;

#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
#else
  unsigned int last_mcon_recv_idx;
#endif

  __device__ MessageIterator(const unsigned int vertex_index) {
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
    // In and only in this case d_vcon.d_in_edge_count has not been accumulated.

    // Since the vertexes are sorted according to their number of out edges.
    mcon_recv_idx =
        (d_vcon.d_in_edge_count[vertex_index] == 0 ? ~0U : vertex_index);

#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    while (mcon_recv_idx != ~0U && !d_mcon_recv.d_is_full[mcon_recv_idx]) {
      mcon_recv_idx = d_auxiliary.d_in_msg_next[mcon_recv_idx];
    }
#endif

#else
    last_mcon_recv_idx = d_vcon.d_in_edge_count[vertex_index];

    mcon_recv_idx =
        (vertex_index == 0 ? 0 : d_vcon.d_in_edge_count[vertex_index - 1]);

#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    // The order of condition expressions in 'while' could not be changed!
    while (mcon_recv_idx != last_mcon_recv_idx
           && !d_mcon_recv.d_is_full[mcon_recv_idx]) {
      ++mcon_recv_idx;
    }
#endif

#endif
  }

  /***************************** api functions ********************************/
  __device__ bool Done() {
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
    return mcon_recv_idx == ~0U;
#else
    return mcon_recv_idx == last_mcon_recv_idx;
#endif
  }

  __device__ void Next() {
    // TODO(laigd): We may need some protection to avoid user invoke this
    // function after all messages have been read?

#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
    // In and only in this case d_vcon.d_in_edge_count has not been accumulated.

#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    do {
#endif
      mcon_recv_idx = d_auxiliary.d_in_msg_next[mcon_recv_idx];
#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
      // The order of condition expressions in 'while' could not be changed!
    } while (mcon_recv_idx != ~0U && !d_mcon_recv.d_is_full[mcon_recv_idx]);
#endif

#else

#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    do {
#endif
      ++mcon_recv_idx;
#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
    } while (
        // The order of condition expressions in 'while' could not be changed!
        mcon_recv_idx != last_mcon_recv_idx
        && !d_mcon_recv.d_is_full[mcon_recv_idx]);
#endif

#endif
  }

  /*************************** system members' get ****************************/
  __device__ unsigned int get_from() const {
    return d_auxiliary.d_in_msg_from[mcon_recv_idx];
  }

  /************************ user defined members' get *************************/
  //// TODO(laigd): add user defined members' GET
$$M[[__device__ <GP_TYPE> get_<GP_NAME>() const { return d_mcon_recv.d_<GP_NAME>[mcon_recv_idx]; }]]
};

struct Vertex {
  // pointing to each member of vcon
  unsigned int index;

  __device__ Vertex(const unsigned int idx) : index(idx) {
  }

  /***************************** api functions ********************************/
  // set it.vid to id[index] and it.e.index to out_edge_count[index - 1] or 0
  // (if index == 0) and returns it.
  __device__ OutEdgeIterator GetOutEdgeIterator() {
    return OutEdgeIterator(index);
  }

  // set alive to 1
  __device__ void KeepAlive() {
    d_gpu_status.alive = true;
  }

  __device__ unsigned int SuperStep() {
    return d_gpu_status.superstep;
  }

#include "user_compute.h"

  /*************************** global members' get/set ************************/
  __device__ unsigned int get_num_vertex() const {
    return d_global.d_num_vertex;
  }

  __device__ unsigned int get_num_edge() const {
    return d_global.d_num_edge;
  }

  //// TODO(laigd): add user defined members' GET
$$G[[__device__ <GP_TYPE> get_<GP_NAME>() const { return d_global.d_<GP_NAME>; }]]

  /*************************** system members' get/set ************************/
  __device__ unsigned int get_id() const {
    return d_vcon.d_id[index];
  }

  __device__ unsigned int get_in_edge_count() const {
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
    // In and only in this case d_vcon.d_in_edge_count has not been accumulated.
    return d_vcon.d_in_edge_count[index];
#else
    unsigned int p = (index == 0 ? 0 : d_vcon.d_in_edge_count[index - 1]);
    return d_vcon.d_in_edge_count[index] - p;
#endif
  }

  __device__ unsigned int get_out_edge_count() const {
    unsigned int p = (index == 0 ? 0 : d_vcon.d_out_edge_count[index - 1]);
    return d_vcon.d_out_edge_count[index] - p;
  }

  /************************ user defined members' get/set *********************/
  //// TODO(laigd): add user defined members' GET/SET
$$V[[__device__ <GP_TYPE> get_<GP_NAME>() const { return d_vcon.d_<GP_NAME>[index]; }]]
$$V_OUT[[__device__ void set_<GP_NAME>(const <GP_TYPE> &value) { d_vcon.d_<GP_NAME>[index] = value; }]]
};

#endif
