// *****************************************************************************
// Filename:    graph_reader.cc
// Date:        2012-12-10 00:03
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "graph_reader.h"

#include "config.h"
#include "generated_io_data_types.h"
#include "rand_util.h"

GraphReader::GraphReader(const Config *config, const unsigned int in_reader_id)
    : conf(config),
      reader_id(in_reader_id),
      num_vertex(config->GetNumVertexForRandomGraph()),
      num_edge(config->GetNumEdgeForRandomGraph()) {
}

GraphReader::~GraphReader() {
}

void GraphReader::RandGlobal(IoGlobal *global) {
  global->num_vertex = num_vertex;
  RandUtil::SetNumVertexOnce(num_vertex);  // See notations in graph_reader.h
  global->num_edge = num_edge;
  IoGlobal::Rand(global);
}
