__device__ void Compute(MessageIterator* msgs) {
  unsigned int phase = SuperStep() % 4;
  if (phase == 0) {
    if (get_left() && get_match_id() == ~0u) { // Left vertex not yet matched.
      for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
        Message msg(*it);
        msg.Send();
      }
    }
  } else if (phase == 1) {
    if (!get_left() && get_match_id() == ~0u) { // Right vertex not yet matched.
      if (!msgs->Done()) {
        // Choose the first id for simple.
        unsigned int match_id = msgs->get_from();
        for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
          if (it->get_to() == match_id) {
            Message msg(*it);
            msg.Send();
            break;
          }
        }
      }
    }
  } else if (phase == 2) {
    if (get_left() && get_match_id() == ~0u) { // Left vertex not yet matched.
      if (!msgs->Done()) {
        // Choose the first id for simple.
        unsigned int match_id = msgs->get_from();
        set_match_id(match_id);

        for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
          if (it->get_to() == match_id) {
            Message msg(*it);
            msg.Send();
            break;
          }
        }
      }
    }
  } else {
    if (!get_left() && get_match_id() == ~0u) { // Right vertex not yet matched.
      if (!msgs->Done()) set_match_id(msgs->get_from());

      // We should notify the system that a cycle is done but the iteration of
      // superstep is not yet finished.
      KeepAlive();
    }
  }
}
