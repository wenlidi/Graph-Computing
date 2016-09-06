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
  // Since the algorithm implemented in GPU is a randomized algorithm, we could
  // never obtain the exact output, so we plan to empty the cpu algorithm here.
}
