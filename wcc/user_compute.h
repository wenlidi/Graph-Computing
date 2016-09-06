__device__ void Compute(MessageIterator* msgs) {
	
//	if(get_id() < get_wcclevel()){
//		set_wcclevel(get_id());
//	}
	
    unsigned int curlevel = get_wcclevel();
//  minlevel_from_msg = get_id();
//  unsigned int mindist = (get_id() == get_source() ? 0 : ~0U);
//  unsigned int pre = ~0U;
    unsigned int minlevel_from_msg = get_id(); 
  for (; !msgs->Done(); msgs->Next()) {
    if (msgs->get_level() < minlevel_from_msg) {
        minlevel_from_msg = msgs->get_level();
    }
  }
  
  if (minlevel_from_msg < curlevel) {
    set_wcclevel(minlevel_from_msg);
    for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
      Message msg(*it);
      msg.set_level(minlevel_from_msg);
      msg.Send();
    }
  }
}
