// *****************************************************************************
// Filename:    random_graph.cc
// Date:        2013-02-26 19:38
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "random_graph.h"

#include <cmath>
#include <iostream>
#include <map>

using std::cerr;
using std::cout;
using std::endl;
using std::map;

RandomGraph::RandomGraph()
    : n(0),
      in_edge_count(NULL),
      out_edge_count(NULL),
      m(0),
      // weight(NULL),
      start(NULL),
      end(NULL) {
}

RandomGraph::~RandomGraph() {
  if (in_edge_count != NULL) delete[] in_edge_count;
  if (out_edge_count != NULL) delete[] out_edge_count;
  if (start != NULL) delete[] start;
  if (end != NULL) delete[] end;
  // if (weight != NULL) delete[] weight;
}

void RandomGraph::OutputStatistics(const RandomGraph *g) {
  map<unsigned int, int> hist_in, hist_out;
  for (unsigned int i = 0; i < g->n; ++i) {
    ++hist_in[g->in_edge_count[i]];
    ++hist_out[g->out_edge_count[i]];
  }
  double expectation = g->m / (double)g->n;

  cout << "In edge distribution "
       << "(number of in-edges X : number of vertexes Y who has X in-edges):"
       << endl;
  double var_in = 0;
  for (map<unsigned int, int>::iterator it = hist_in.begin();
       it != hist_in.end(); ++it) {
    double diff = it->first - expectation;
    var_in += diff * diff * it->second;
    cout << it->first << ":" << it->second << ", ";
  }
  cout << endl;
  cout << "Max number of in-edge: " << hist_in.rbegin()->first << endl;
  cout << "Standard deviation: " << std::sqrt(var_in / g->n) << endl;

  cout << "Out edge distribution "
       << "(number of out-edges X : number of vertexes Y who has X out-edges):"
       << endl;
  double var_out = 0;
  for (map<unsigned int, int>::iterator it = hist_out.begin();
       it != hist_out.end(); ++it) {
    double diff = it->first - expectation;
    var_out += diff * diff * it->second;
    cout << it->first << ":" << it->second << ", ";
  }
  cout << endl;
  cout << "Max number of out-edge: " << hist_out.rbegin()->first << endl;
  cout << "Standard deviation: " << std::sqrt(var_out / g->n) << endl;
}
