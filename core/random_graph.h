// *****************************************************************************
// Filename:    random_graph.h
// Date:        2013-02-26 19:26
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RANDOM_GRAPH_H_
#define RANDOM_GRAPH_H_

struct RandomGraph {
  /* No. of vertices, represented by n */
  unsigned int n;
  unsigned int *in_edge_count;
  unsigned int *out_edge_count;

  /* No. of edges, represented by m */
  unsigned int m;

  /* Arrays of size 'm' storing the edge information
   * A directed edge 'e' (0 <= e < m) from start[e] to end[e]
   * had an integer weight w[e] */
  unsigned int *start;
  unsigned int *end;

  // unsigned int *weight;

  RandomGraph();

  ~RandomGraph();

  static void OutputStatistics(const RandomGraph *g);
};

#endif
