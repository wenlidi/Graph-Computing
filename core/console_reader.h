// *****************************************************************************
// Filename:    console_reader.h
// Date:        2013-03-25 15:23
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef CONSOLE_READER_H_
#define CONSOLE_READER_H_

#include "graph_reader.h"

class Config;
class SharedData;
struct IoGlobal;

class ConsoleReader : public GraphReader {
 public:

  ConsoleReader(const Config *conf, const unsigned int in_reader_id);

  // Format:
  // num_vertex
  // num_edge
  virtual void ReadGlobal(IoGlobal *global);

  // Format:
  // num_vertex_in_current_shard
  // vid_1 in_edge_count_1 out_edge_count_1 user_defined_member_a_1 ...
  // ...
  // vid_n in_edge_count_n out_edge_count_n user_defined_member_a_n ...
  virtual void ReadVertexContent(SharedData *shared_data);

  // Format:
  // num_edge_in_current_shard
  // from_1 to_1 user_defined_member_a_1 user_defined_member_b_1 ...
  // ...
  // from_m to_m user_defined_member_a_m user_defined_member_b_m ...
  virtual void ReadEdgeContent(SharedData *shared_data);

};

#endif
