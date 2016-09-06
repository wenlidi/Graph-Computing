// *****************************************************************************
// Filename:    host_graph.cu
// Date:        2012-10-23 16:31
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "host_graph.h"

#include <algorithm>
#include <iostream>
#include <vector>

#include "constants.h"
#include "generated_io_data_types.h"
#include "host_graph_data_types.h"
#include "cpu_algorithm.h"
#include "config.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> [HostGraph]: "
#endif

#ifdef LAMBDA_PROFILING
#include "singleton.h"
#include "profiler.h"
#endif

using std::vector;
using std::cout;
using std::endl;

HostGraph::HostGraph(const Config *c)
    : conf(c),
      global(),
      vertex_vec(),
      edge_vec() {
}

void HostGraph::SetGlobal(const IoGlobal &g) {
  global.Set(g);
  vertex_vec.reserve(global.num_vertex);
  edge_vec.reserve(global.num_edge);
}

void HostGraph::AddVertex(const IoVertex &v) {
  vertex_vec.resize(vertex_vec.size() + 1);
  vertex_vec.back().Set(v);
}

void HostGraph::AddEdge(const IoEdge &e) {
  edge_vec.resize(edge_vec.size() + 1);
  edge_vec.back().Set(e);
}

void HostGraph::RunAlgorithm() {
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StartTimer("CPU prepare");
#endif
  Prepare();
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StopTimer("CPU prepare");
#endif

#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StartTimer("CPU run algorithm");
#endif
  CpuAlgorithm(conf, global, vertex_vec, edge_vec);
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StopTimer("CPU run algorithm");
#endif
}

void HostGraph::Prepare() {
  if (global.num_vertex != vertex_vec.size()) {
    cout << "HostGraph::FinishedReadingDataAndPrepare error: "
         << "number of vertexes do not match! "
         << "global.num_vertex: " << global.num_vertex
         << endl;
    exit(1);
  }
  if (global.num_edge != edge_vec.size()) {
    cout << "HostGraph::FinishedReadingDataAndPrepare error: "
         << "number of edges do not match! "
         << "global.num_edge: " << global.num_edge
         << endl;
    exit(1);
  }

  std::sort(vertex_vec.begin(), vertex_vec.end());
  std::sort(edge_vec.begin(), edge_vec.end());
  for (unsigned int i = 0, sum = 0; i < global.num_vertex; ++i) {
    vertex_vec[i].sum_out_edge_count += sum;
    sum = vertex_vec[i].sum_out_edge_count;
  }
}

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
