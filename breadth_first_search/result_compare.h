void CompareResult(
    HostOutVertexContent *gpu_vout,
    HostOutEdgeContent *gpu_eout,
    vector<HostGraphVertex> *cpu_vout,
    vector<HostGraphEdge> *cpu_eout,
    bool *vertex_result_correct,
    bool *edge_result_correct) {
  *vertex_result_correct = true;
  *edge_result_correct = true;

  gpu_vout->SortById();
  for (unsigned int i = 0; i < gpu_vout->Size(); ++i) {
    if (gpu_vout->id[i] != (*cpu_vout)[i].id) {
      *vertex_result_correct = false;
      return;
    }
    if (gpu_vout->level[i] != (*cpu_vout)[i].level) {
      *vertex_result_correct = false;
      return;
    }
  }
}
