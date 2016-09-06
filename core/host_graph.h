// *****************************************************************************
// Filename:    host_graph.h
// Date:        2012-12-29 12:51
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef HOST_GRAPH_H_
#define HOST_GRAPH_H_

#include <vector>

#include "generated_io_data_types.h"
#include "host_graph_data_types.h"
#include "config.h"

using std::vector;

class HostGraph {
 public:

  HostGraph(const Config *c);

  void SetGlobal(const IoGlobal &g);

  void AddVertex(const IoVertex &v);

  void AddEdge(const IoEdge &e);

  void RunAlgorithm();

  vector<HostGraphVertex>* GetVertexVector() {
    return &vertex_vec;
  }

  vector<HostGraphEdge>* GetEdgeVector() {
    return &edge_vec;
  }

 private:

  const Config *conf;

  HostGraphGlobal global;

  vector<HostGraphVertex> vertex_vec;

  vector<HostGraphEdge> edge_vec;

  // Build indexes.
  void Prepare();

};

#endif
