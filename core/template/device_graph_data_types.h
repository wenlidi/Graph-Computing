// *****************************************************************************
// Filename:    device_graph_data_types.h
// Date:        2012-12-06 15:28
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GENERATED_GRAPH_DATA_TYPES_H_
#define GENERATED_GRAPH_DATA_TYPES_H_

struct Global {
  unsigned int d_num_vertex;
  unsigned int d_num_edge;

  //// TODO(laigd): add user defined members
$$G[[<GP_TYPE> d_<GP_NAME>;]]
};

struct VertexContent {
  unsigned int d_size;
  unsigned int *d_id;

  // After 'GPUStorageManager::BuildIndexes', both counts are prefix sums.
  unsigned int *d_in_edge_count;
  unsigned int *d_out_edge_count;

  //// TODO(laigd): add user defined members
$$V[[<GP_TYPE> *d_<GP_NAME>;]]
};

struct EdgeContent {
  unsigned int d_size;

  // @from could not be discarded after index built (Step 3) because in the
  // future we may need it to run user defined aggregation functions.
  unsigned int *d_from;
  unsigned int *d_to;

  //// TODO(laigd): add user defined members
$$E[[<GP_TYPE> *d_<GP_NAME>;]]
};

struct MessageContent {
#ifdef LAMBDA_SHARE_ONE_MESSAGE_ARRAY
  unsigned int d_space_size;
  unsigned int *d_space;
#endif

  unsigned int d_size;

#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
  bool *d_is_full;
#endif

  //// TODO(laigd): add user defined members
$$M[[<GP_TYPE> *d_<GP_NAME>;]]
};

struct AuxiliaryDeviceData {
  unsigned int d_num_gpus;

  // Size: number of in edges.
  unsigned int *d_in_msg_from;
#if defined(LAMBDA_SORT_VERTEX_BY_IN_EDGE_COUNT) \
    && defined(LAMBDA_IN_EDGE_COALESCED_MEMORY_ACCESS)
  unsigned int *d_in_msg_next;
#endif

  // Size: number of out edges.
  unsigned int *d_out_edge_in_msg_map;
};

#endif
