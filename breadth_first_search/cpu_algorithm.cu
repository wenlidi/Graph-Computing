#include "cpu_algorithm.h"

#include <algorithm>
#include <vector>

#include "host_graph_data_types.h"

using std::vector;

void CpuAlgorithm(
    const Config *conf,
    HostGraphGlobal &global,
    vector<HostGraphVertex> &vertex_vec,
    vector<HostGraphEdge> &edge_vec) {
  vector<unsigned int> queue;
  vector<bool> vst(global.num_vertex);
  unsigned int max_level = 0;

  vertex_vec[global.root].level = 0;
  queue.push_back(global.root);
  vst[global.root] = true;

  for (unsigned int i = 0; i < queue.size(); ++i) {
    unsigned int vid = queue[i];
    const unsigned int level = vertex_vec[vid].level + 1;

    const unsigned int begin =
      (vid == 0 ? 0 : vertex_vec[vid - 1].sum_out_edge_count);
    const unsigned int end = vertex_vec[vid].sum_out_edge_count;

    for (unsigned int k = begin; k < end; ++k) {
      unsigned int to = edge_vec[k].to;
      if (!vst[to]) {
        if (level > max_level) max_level = level;
        vertex_vec[to].level = level;
        queue.push_back(to);
        vst[to] = true;
      }
    }
  }
}
