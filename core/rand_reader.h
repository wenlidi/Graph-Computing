// *****************************************************************************
// Filename:    rand_reader.h
// Date:        2013-02-26 19:17
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RANDOM_READER_H_
#define RANDOM_READER_H_

#include "random_graph_reader.h"

class Config;
class SharedData;
class RandomGraph;
struct IoGlobal;

class RandReader : public RandomGraphReader {
 public:

  RandReader(const Config *conf, const unsigned int in_reader_id);

  virtual void GenGraph(IoGlobal *global);
};

#endif
