__device__ float Compute(MessageIterator* msgs) {
  if (SuperStep() > 0) {
    float sum = 0;
    for (; !msgs->Done(); msgs->Next()) {
      sum += msgs->get_rank();
    }
    set_rank(sum);
  }

  float averaged = get_rank() / get_out_edge_count();
  return averaged;
  
}
