// *****************************************************************************
// Filename:    rmat_reader.h
// Date:        2013-02-26 12:36
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RMAT_READER_H_
#define RMAT_READER_H_

#include "random_graph_reader.h"

class Config;
class SharedData;
struct IoGlobal;
class RandomGraph;

class RMatReader : public RandomGraphReader {
 public:

  RMatReader(const Config *conf, const unsigned int in_reader_id);

  virtual void GenGraph(IoGlobal *global);
};

#endif
