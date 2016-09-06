__device__ void Compute(MessageIterator* msgs) {
  unsigned level = (get_id() == get_root() ? 0 : get_level());
  if (!msgs->Done()) level = SuperStep();
  if (level < get_level()) {
    set_level(level);
    for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
      Message msg(*it);
      msg.Send();
    }
  }
}
