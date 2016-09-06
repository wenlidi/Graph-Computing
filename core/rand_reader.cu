// *****************************************************************************
// Filename:    rand_reader.cu
// Date:        2013-02-26 19:36
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "rand_reader.h"

#include "config.h"
#include "rand_generator.h"
#include "generated_io_data_types.h"
#include "random_graph.h"

RandReader::RandReader(const Config *conf, const unsigned int in_reader_id)
    : RandomGraphReader(conf, in_reader_id) {
}

void RandReader::GenGraph(IoGlobal *global) {
  RandGenerator gen(global->num_vertex, global->num_edge, false);
  gen.GenGraph(graph);
}
