// *****************************************************************************
// Filename:    cpu_algorithm.cc
// Date:        2013-01-01 17:35
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "cpu_algorithm.h"

#include <algorithm>
#include <vector>
#include <iostream>

#include "adjustable_heap.h"
#include "config.h"
#include "host_graph_data_types.h"

using std::vector;
using std::cout;
using std::cin;
using std::endl;

struct DijkNode {
  unsigned int v;
  unsigned int pre;
  unsigned int c;

  // Add data member @heap_position.
  ADD_ADJUSTABLE_HEAP_MEMBER();

  DijkNode() : v(~0U), pre(~0U), c(~0U), heap_position(kInvalidPos) {
  }

  bool operator<(const DijkNode &other) const {
    return c < other.c;
  }
};

void CpuAlgorithm(
    const Config *conf,
    HostGraphGlobal &global,
    vector<HostGraphVertex> &vertex_vec,
    vector<HostGraphEdge> &edge_vec) {
	cout<<" CPU Algorithm been executed! "<<endl;
  vector<DijkNode> heap_data(global.num_vertex);
  heap_data[global.source].c = 0;
  for (unsigned int i = 0; i < global.num_vertex; ++i) {
    heap_data[i].v = i;
  }
  AdjustableHeap<DijkNode> heap(heap_data.begin(), heap_data.end());

  while (heap.Size() > 0) {
    const DijkNode &top = *(heap.Pop());
    if (top.c == ~0U) break;

    const unsigned int begin =
        (top.v == 0 ? 0 : vertex_vec[top.v - 1].sum_out_edge_count);
    const unsigned int end = vertex_vec[top.v].sum_out_edge_count;
    for (unsigned int j = begin; j < end; ++j) {
      unsigned int jt = edge_vec[j].to, jc = edge_vec[j].weight;
      if (heap_data[jt].c > top.c + jc) {
        heap_data[jt].c = top.c + jc;
        heap_data[jt].pre = top.v;
        heap.UpHeap(heap_data[jt].heap_position);
      }
    }
  }

  for (unsigned int i = 0; i < global.num_vertex; ++i) {
    vertex_vec[i].dist = heap_data[i].c;
    vertex_vec[i].pre = heap_data[i].pre;
  }
}
