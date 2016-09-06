// *****************************************************************************
// Filename:    host_graph_data_types.h
// Date:        2012-12-22 18:10
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: This file contains data structures used by HostGraph to run
//              corresponding algorithms on CPU.
// *****************************************************************************

#ifndef GENERATED_GRAPH_DATA_TYPES_ON_HOST_H_
#define GENERATED_GRAPH_DATA_TYPES_ON_HOST_H_

#include <iostream>

#include "generated_io_data_types.h"

using std::ostream;
using std::endl;

struct HostGraphGlobal {
  unsigned int num_vertex;
  unsigned int num_edge;
  //// TODO(laigd): add user defined members
$$G[[<GP_TYPE> <GP_NAME>;]]

  void Set(const IoGlobal &g) {
    num_vertex = g.num_vertex;
    num_edge = g.num_edge;
    //// TODO(laigd): add user defined members
$$G[[<GP_NAME> = g.<GP_NAME>;]]
  }
};

struct HostGraphVertex {
  unsigned int id;
  unsigned int sum_out_edge_count;
  //// TODO(laigd): add user defined members
$$V[[<GP_TYPE> <GP_NAME>;]]

  void Set(const IoVertex &v) {
    id = v.id;
    sum_out_edge_count = v.out_edge_count;
    //// TODO(laigd): add user defined members
$$V_IN[[<GP_NAME> = v.<GP_NAME>;]]
$$V_OUT[[<GP_NAME> = <GP_INIT_VALUE>;]]
  }

  void Write(ostream &out) const {
    out << id
        //// TODO(laigd): add user defined 'out' members
$$V[[<< ", " << <GP_NAME>]]
        << endl;
  }

  bool operator<(const HostGraphVertex &rhs) const {
    return id < rhs.id;
  }
};

struct HostGraphEdge {
  unsigned int from;
  unsigned int to;
  //// TODO(laigd): add user defined members
$$E[[<GP_TYPE> <GP_NAME>;]]

  void Set(const IoEdge &e) {
    from = e.from;
    to = e.to;
    //// TODO(laigd): add user defined members
$$E_IN[[<GP_NAME> = e.<GP_NAME>;]]
$$E_OUT[[<GP_NAME> = <GP_INIT_VALUE>;]]
  }

  void Write(ostream &out) const {
    out << from << ", " << to
        //// TODO(laigd): add user defined 'out' members
$$E[[<< ", " << <GP_NAME>]]
        << endl;
  }

  bool operator<(const HostGraphEdge &rhs) const {
    return (from == rhs.from ? to < rhs.to : from < rhs.from);
  }
};

#endif
