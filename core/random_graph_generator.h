// *****************************************************************************
// Filename:    random_graph_generator.h
// Date:        2013-03-05 15:29
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RANDOM_GRAPH_GENERATOR_H_
#define RANDOM_GRAPH_GENERATOR_H_

class RandomGraph;

class RandomGraphGenerator {
 public:

  RandomGraphGenerator(
      const unsigned int num_vertex,
      const unsigned int num_edge,
      const bool self_loops_or_not)
      : n(num_vertex),
        m(num_edge),
        self_loops(self_loops_or_not) {
  }

  virtual ~RandomGraphGenerator() {
  }

  virtual void GenGraph(RandomGraph *g) = 0;

 protected:

  unsigned int n, m;
  bool self_loops;
};

#endif
