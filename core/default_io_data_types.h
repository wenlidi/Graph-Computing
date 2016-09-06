// *****************************************************************************
// Filename:    default_io_data_type.h
// Date:        2012-12-06 14:02
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef DEFAULT_IO_DATA_TYPE_H_
#define DEFAULT_IO_DATA_TYPE_H_

// Only members of Vertex, Edge and Global can be marked as 'in' or 'out' or 'io'
struct Global {
  in unsigned int num_vertex;
  in unsigned int num_edge;
};

struct Vertex {
  in unsigned int id;
  in unsigned int in_edge_count;
  in unsigned int out_edge_count;
};

struct Edge {
  in unsigned int from;
  in unsigned int to;
};

struct Message {
  unsigned int from;
  unsigned int to;
};

#endif
