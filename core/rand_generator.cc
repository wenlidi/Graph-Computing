// *****************************************************************************
// Filename:    rand_generator.cc
// Date:        2013-03-06 12:19
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "rand_generator.h"

#include <cstdlib>

#include "random_graph.h"
#include "random_graph_generator.h"
#include "sprng.h"

RandGenerator::RandGenerator(
    const unsigned int num_vertex,
    const unsigned int num_edge,
    const bool self_loops_or_not)
    : RandomGraphGenerator(num_vertex, num_edge, self_loops_or_not) {
}

void RandGenerator::GenGraph(RandomGraph *g) {
  unsigned int *in_edge_count = new unsigned int[n];
  unsigned int *out_edge_count = new unsigned int[n];
  unsigned int *start_vertex = new unsigned int[m];
  unsigned int *end_vertex = new unsigned int[m];
  // unsigned int *weight = new unsigned int[m];

  for (unsigned int i = 0; i < n; ++i) {
    in_edge_count[i] = 0;
    out_edge_count[i] = 0;
  }

  int *stream1 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED1, SPRNG_DEFAULT);
  int *stream2 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED2, SPRNG_DEFAULT);

  for (unsigned int i = 0; i < m; ++i) {
    unsigned int u = (unsigned int) isprng(stream1) % n;
    unsigned int v = (unsigned int) isprng(stream1) % n;
    if ((u == v) && !self_loops) {
      i--;
      continue;
    }

    // weight[i] = 0 + (unsigned int) (100 - 0) * sprng(stream2);

    start_vertex[i] = u;
    ++out_edge_count[u];

    end_vertex[i] = v;
    ++in_edge_count[v];
  }
  free(stream1);
  free(stream2);

  g->n = n;
  g->in_edge_count = in_edge_count;
  g->out_edge_count = out_edge_count;
  g->m = m;
  g->start = start_vertex;
  g->end = end_vertex;
  // g->weight = weight;
#ifdef LAMBDA_GRAPH_STATISTICS
  RandomGraph::OutputStatistics(g);
#endif
}
