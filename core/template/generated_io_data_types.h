// *****************************************************************************
// Filename:    generated_io_data_types.h
// Date:        2012-12-11 16:49
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: This file contains data structures used for input operations.
//              Each of these structures only contain user defined 'in' member.
// *****************************************************************************

#ifndef GENERATED_IO_DATA_TYPES_H_
#define GENERATED_IO_DATA_TYPES_H_

#include <istream>

#include "rand_util.h"

using std::istream;

struct IoGlobal {
  unsigned int num_vertex;
  unsigned int num_edge;
  //// TODO(laigd): add user defined 'in' members
$$G[[<GP_TYPE> <GP_NAME>;]]

  // Read default and user defined 'in' members
  static void Read(istream &in, IoGlobal *global) {
    in >> global->num_vertex;
    in >> global->num_edge;
    //// TODO(laigd): add code to read user defined 'in' members
$$G[[in >> global-><GP_NAME>;]]
  }

  // Only generate user defined 'in' members
  static void Rand(IoGlobal *global) {
    //// TODO(laigd): add code to generate user defined 'in' members
$$G[[global-><GP_NAME> = <GP_RAND_VALUE>;]]
  }
};

struct IoVertex {
  unsigned int id;
  unsigned int in_edge_count;
  unsigned int out_edge_count;
  //// TODO(laigd): add user defined 'in' members
$$V_IN[[<GP_TYPE> <GP_NAME>;]]

  static void Read(istream &in, IoVertex *vertex) {
    in >> vertex->id;
    in >> vertex->in_edge_count;
    in >> vertex->out_edge_count;
    //// TODO(laigd): add code to read user defined 'in' members
$$V_IN[[in >> vertex-><GP_NAME>;]]
  }

  static void Rand(IoVertex *vertex) {
    //// TODO(laigd): add code to generate user defined 'in' members
$$V_IN[[vertex-><GP_NAME> = <GP_RAND_VALUE>;]]
  }
};

struct IoEdge {
  unsigned int from;
  unsigned int to;
  //// TODO(laigd): add user defined 'in' members
$$E_IN[[<GP_TYPE> <GP_NAME>;]]

  static void Read(istream &in, IoEdge *edge) {
    in >> edge->from;
    in >> edge->to;
    //// TODO(laigd): add code to read user defined 'in' members
$$E_IN[[in >> edge-><GP_NAME>;]]
  }

  static void Rand(IoEdge *edge) {
    //// TODO(laigd): add code to generate user defined 'in' members
$$E_IN[[edge-><GP_NAME> = <GP_RAND_VALUE>;]]
  }
};

#endif
