// *****************************************************************************
// Filename:    simple_reader.h
// Date:        2012-12-09 22:24
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef UNIFORM_READER_H_
#define UNIFORM_READER_H_

#include "graph_reader.h"

class Config;
class SharedData;
struct IoGlobal;

// We use @reader_id to partition vertexes and edges to different reader.
class SimpleReader : public GraphReader {
 public:

  SimpleReader(const Config *conf, const unsigned int in_reader_id);

  virtual void ReadGlobal(IoGlobal *global);

  virtual void ReadVertexContent(SharedData *shared_data);

  virtual void ReadEdgeContent(SharedData *shared_data);

 private:

  unsigned int num_vertex_for_current_shard;

  unsigned int num_edge_for_current_shard;

  unsigned int vertex_id_offset;

  void Init();
};

#endif
