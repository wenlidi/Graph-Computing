__device__ int Compute(MessageIterator* msgs) {
	
    unsigned int curlevel = get_wcclevel();
    unsigned int minlevel_from_msg = get_id(); 
  for (; !msgs->Done(); msgs->Next()) {
    if (msgs->get_level() < minlevel_from_msg) {
        minlevel_from_msg = msgs->get_level();
    }
  }
  
  if (minlevel_from_msg < curlevel) {
 
       set_wcclevel(minlevel_from_msg);
	   return minlevel_from_msg;
      /* 
      for (OutEdgeIterator it = GetOutEdgeIterator(); !it.Done(); it.Next()) {
      Message msg(*it);
      msg.set_level(minlevel_from_msg);
      msg.Send();
    }
	*/
	}
	return -1;
  
}
