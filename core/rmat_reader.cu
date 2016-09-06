// *****************************************************************************
// Filename:    rmat_reader.cu
// Date:        2013-02-26 13:00
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "rmat_reader.h"

#include "config.h"
#include "rmat_generator.h"
#include "generated_io_data_types.h"
#include "random_graph.h"

RMatReader::RMatReader(const Config *conf, const unsigned int in_reader_id)
    : RandomGraphReader(conf, in_reader_id) {
}

void RMatReader::GenGraph(IoGlobal *global) {
  RMatGenerator gen(global->num_vertex, global->num_edge, false);
  gen.GenGraph(graph);
}
