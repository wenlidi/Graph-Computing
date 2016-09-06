// *****************************************************************************
// Filename:    graph_reader.h
// Date:        2012-12-09 22:23
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GRAPH_READER_H_
#define GRAPH_READER_H_

class SharedData;
class Config;
struct IoGlobal;

class GraphReader {
 public:

  GraphReader(const Config *config, const unsigned int in_reader_id);

  virtual ~GraphReader();

  // If the graph is not obtained from file but generated randomly, the
  // derivative class should implement this function so that RandUtil knows the
  // number of vertexes just and only just after the reader gets it, (i.e. by
  // invoking RandUtil::SetNumVertexOnce), or alternatively, we can just invoke
  // the following protected function RandGlobal to do the job.
  virtual void ReadGlobal(IoGlobal *global) = 0;

  virtual void ReadVertexContent(SharedData *shared_data) = 0;

  virtual void ReadEdgeContent(SharedData *shared_data) = 0;

 protected:

  const Config *conf;

  // Starts from 0.
  const unsigned int reader_id;

  unsigned int num_vertex;

  unsigned int num_edge;

  void RandGlobal(IoGlobal *global);
};

#endif
