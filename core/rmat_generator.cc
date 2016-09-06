// *****************************************************************************
// Filename:    rmat_generator.cc
// Date:        2013-03-06 12:11
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "rmat_generator.h"

#include <cstdlib>

#include "random_graph.h"
#include "random_graph_generator.h"
#include "sprng.h"

RMatGenerator::RMatGenerator(
    const unsigned int num_vertex,
    const unsigned int num_edge,
    const bool self_loops_or_not)
    : RandomGraphGenerator(num_vertex, num_edge, self_loops_or_not) {
  a = 0.45;
  b = 0.15;
  c = 0.15;
  d = 0.25;
}

void RMatGenerator::GenGraph(RandomGraph *g) {
  // Initialize SPRNG
  int *stream1 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED1, SPRNG_DEFAULT);
  int *stream2 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED2, SPRNG_DEFAULT);
  int *stream3 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED3, SPRNG_DEFAULT);
  int *stream4 = init_sprng(SPRNG_CMRG, 0, 1, SPRNG_SEED4, SPRNG_DEFAULT);

  // Generate edges as per the graph model and user options
  unsigned int *in_edge_count = new unsigned int[n];
  unsigned int *out_edge_count = new unsigned int[n];
  unsigned int *start_vertex = new unsigned int[m];
  unsigned int *end_vertex = new unsigned int[m];
  // unsigned int *weight = new unsigned int[m];

  for (unsigned int i = 0; i < n; ++i) {
    in_edge_count[i] = 0;
    out_edge_count[i] = 0;
  }

  double a0 = a, b0 = b, c0 = c, d0 = d;
  for (unsigned int i = 0; i < m; ++i) {
    a = a0; b = b0; c = c0; d = d0;
    int u = 1, v = 1;

    int step = n/2;
    while (step >= 1) {
      ChoosePartition(&u, &v, step, stream1);
      step = step/2;
      VaryParams(&a, &b, &c, &d, stream4, stream3);
    }

    /* Create edge [u-1, v-1] */
    if ((!self_loops) && (u == v)) {
      i--;
      continue;
    }

    // weight[i] = 0 + (unsigned int) (100 - 0) * sprng(stream2);

    start_vertex[i] = u-1;
    ++out_edge_count[u-1];

    end_vertex[i] = v-1;
    ++in_edge_count[v-1];
  }
  free(stream1);
  free(stream2);
  free(stream3);
  free(stream4);

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

void RMatGenerator::ChoosePartition(int *u, int* v, int step, int *stream) {
  double p;
  p = sprng(stream);
  if (p < a) {
    /* Do nothing */
  } else if ((a < p) && (p < a+b)) {
    *v = *v + step;
  } else if ((a+b < p) && (p < a+b+c)) {
    *u = *u + step;
  } else if ((a+b+c < p) && (p < a+b+c+d)) {
    *u = *u + step;
    *v = *v + step;
  }
}

void RMatGenerator::VaryParams(
    double* a, double* b, double* c, double* d,
    int *stream_a, int *stream_b) {
  /* Allow a max. of 5% variation */
  double v= 0.05;

  if (sprng(stream_a) > 0.5) {
    *a += *a * v * sprng(stream_b);
  } else {
    *a -= *a * v * sprng(stream_b);
  }

  if (sprng(stream_a) > 0.5) {
    *b += *b * v * sprng(stream_b);
  } else {
    *b += *b * v * sprng(stream_b);
  }

  if (sprng(stream_a) > 0.5) {
    *c += *c * v * sprng(stream_b);
  } else {
    *c -= *c * v * sprng(stream_b);
  }

  if (sprng(stream_a) > 0.5) {
    *d += *d * v * sprng(stream_b);
  } else {
    *d -= *d * v * sprng(stream_b);
  }

  double s = *a + *b + *c + *d;
  *a = *a/s;
  *b = *b/s;
  *c = *c/s;
  *d = *d/s;
}
