// *****************************************************************************
// Filename:    rand_generator.h
// Date:        2013-03-06 12:16
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RAND_GENERATOR_H_
#define RAND_GENERATOR_H_

#include "random_graph_generator.h"

class RandGenerator : RandomGraphGenerator {
 public:

  RandGenerator(
      const unsigned int num_vertex,
      const unsigned int num_edge,
      const bool self_loops_or_not);

  virtual void GenGraph(RandomGraph *g);

 private:

  static const long int SPRNG_SEED1 = 1261983;
  static const long int SPRNG_SEED2 = 312198;
};

#endif
