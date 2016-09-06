// *****************************************************************************
// Filename:    random_graph_reader.cu
// Date:        2013-02-26 13:00
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RANDOM_GRAPH_READER_H_
#define RANDOM_GRAPH_READER_H_

#include "graph_reader.h"

class Config;
class SharedData;
class RandomGraph;
struct IoGlobal;

class RandomGraphReader : public GraphReader {
 public:

  RandomGraphReader(const Config *conf, const unsigned int in_reader_id);

  virtual ~RandomGraphReader();

  virtual void GenGraph(IoGlobal *global) = 0;

  virtual void ReadGlobal(IoGlobal *global);

  virtual void ReadVertexContent(SharedData *shared_data);

  virtual void ReadEdgeContent(SharedData *shared_data);

 protected:

  RandomGraph *graph;
};

#endif
