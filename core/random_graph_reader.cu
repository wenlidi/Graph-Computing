// *****************************************************************************
// Filename:    random_graph_reader.cc
// Date:        2013-03-06 09:11
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "random_graph_reader.h"

#include <iostream>

#include "config.h"
#include "graph_reader.h"
#include "generated_io_data_types.h"
#include "shared_data.h"
#include "random_graph.h"
#include "random_graph_generator.h"

using std::cerr;
using std::cin;
using std::cout;
using std::endl;

RandomGraphReader::RandomGraphReader(
    const Config *conf,
    const unsigned int in_reader_id)
    : GraphReader(conf, in_reader_id),
  graph(NULL) {
  if (reader_id != 0) {
    cout << "Only ONE thread is allowed in R-MAT reader!" << endl;
    exit(1);
  }
}

RandomGraphReader::~RandomGraphReader() {
  if (graph != NULL) delete graph;
}

void RandomGraphReader::ReadGlobal(IoGlobal *global) {
  RandGlobal(global);  // TODO
  graph = new RandomGraph();
  GenGraph(global);
}

void RandomGraphReader::ReadVertexContent(SharedData *shared_data) {
  for (unsigned int i = 0; i < graph->n; ++i) {
    IoVertex v;
    v.id = i;
    v.in_edge_count = graph->in_edge_count[i];
    v.out_edge_count = graph->out_edge_count[i];
    IoVertex::Rand(&v);
    shared_data->AddVertex(v);
  }
}

void RandomGraphReader::ReadEdgeContent(SharedData *shared_data) {
  for (unsigned int i = 0; i < graph->m; ++i) {
    IoEdge e;
    e.from = graph->start[i];
    e.to = graph->end[i];
    IoEdge::Rand(&e);  // TODO
    shared_data->AddEdge(e);
  }
}
