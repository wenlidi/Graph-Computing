__device__ int Compute(MessageIterator* msgs) {
  unsigned int mindist = (get_id() == get_source() ? 0 : ~0U);
  unsigned int pre = ~0U;

  for (; !msgs->Done(); msgs->Next()) {
    if (msgs->get_dist() < mindist) {
      mindist = msgs->get_dist();
      pre = msgs->get_from();
    }
  }

  if (mindist < get_dist()) {
    set_dist(mindist);
    set_pre(pre);
    return mindist;
   // sendmessage <<<1,num_of_thread>>>(it,mindist);
    /*
    for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
      Message msg(*it);
      msg.set_dist(mindist + it->get_weight());
      msg.Send();
    }
    */
  }
  return -1;
}

