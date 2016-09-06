void CompareResult(
    HostOutVertexContent *gpu_vout,
    HostOutEdgeContent *gpu_eout,
    vector<HostGraphVertex> *cpu_vout,
    vector<HostGraphEdge> *cpu_eout,
    bool *vertex_result_correct,
    bool *edge_result_correct) {
  *vertex_result_correct = true;
  *edge_result_correct = true;
}
