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
  vector<float> tmp_rank(global.num_vertex);
  for (unsigned int i = 0; i < conf->GetMaxSuperstep(); ++i) {
    if (i > 0) {
      for (unsigned int j = 0; j < global.num_vertex; ++j) {
        vertex_vec[j].rank = tmp_rank[j];
      }
    }

    std::fill(tmp_rank.begin(), tmp_rank.end(), 0);
    for (unsigned int j = 0; j < global.num_vertex; ++j) {
      const unsigned int begin =
          (j == 0 ? 0 : vertex_vec[j - 1].sum_out_edge_count);
      const unsigned int end = vertex_vec[j].sum_out_edge_count;
      float averaged = vertex_vec[j].rank / (end - begin);

      for (unsigned int k = begin; k < end; ++k) {
        tmp_rank[edge_vec[k].to] += averaged;
      }
    }
  }
}
