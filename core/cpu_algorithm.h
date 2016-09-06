// *****************************************************************************
// Filename:    cpu_algorithm.h
// Date:        2013-01-01 17:33
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef CPU_ALGORITHM_H_
#define CPU_ALGORITHM_H_

#include <vector>

#include "host_graph_data_types.h"
#include "config.h"

using std::vector;

void CpuAlgorithm(
    const Config *conf,
    HostGraphGlobal &global,
    vector<HostGraphVertex> &vertex_vec,
    vector<HostGraphEdge> &edge_vec);

#endif
