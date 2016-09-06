__device__ void Compute(MessageIterator* msgs) {
  if (SuperStep() > 0) {
    float sum = 0;
    for (; !msgs->Done(); msgs->Next()) {
      sum += msgs->get_rank();
    }
    set_rank(sum);
  }

  float averaged = get_rank() / get_out_edge_count();
  for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
    Message msg(*it);
    msg.set_rank(averaged);
    msg.Send();
  }
}
