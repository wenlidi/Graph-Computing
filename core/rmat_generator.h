// *****************************************************************************
// Filename:    rmat_generator.h
// Date:        2013-03-06 12:08
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RMAT_GENERATOR_H_
#define RMAT_GENERATOR_H_

#include "random_graph_generator.h"

class RMatGenerator : RandomGraphGenerator {
 public:

  RMatGenerator(
      const unsigned int num_vertex,
      const unsigned int num_edge,
      const bool self_loops_or_not);

  virtual void GenGraph(RandomGraph *g);

 private:

  static const long int SPRNG_SEED1 = 12619830;
  static const long int SPRNG_SEED2 = 31219885;
  static const long int SPRNG_SEED3 = 72824922;
  static const long int SPRNG_SEED4 = 81984016;

  double a, b, c, d;

  void ChoosePartition(int *u, int* v, int step, int *stream);

  void VaryParams(
      double* a, double* b, double* c, double* d,
      int *stream_a, int *stream_b);
};

#endif
