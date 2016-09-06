// *****************************************************************************
// Filename:    single_stream_writer.cc
// Date:        2012-12-26 14:43
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "single_stream_writer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include <string>

#include "device_graph_data_types.h"
#include "host_out_graph_data_types.h"
#include "output_writer.h"
#include "auto_mutex.h"

using std::ostream;
using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::ostream;
using std::string;

namespace {

void WriteOutputHandler(
    HostOutVertexContent *out_vcon,
    HostOutEdgeContent *out_econ,
    ostream *out) {
  out_vcon->SortById();
  out_vcon->Write(*out);
  out_econ->SortByFromTo();
  out_econ->Write(*out);
}

}  // namespace

SingleStreamWriter::SingleStreamWriter(
    Global *g,
    VertexContent *v,
    EdgeContent *e) : OutputWriter(g, v, e) {
}

void SingleStreamWriter::WriteOutput() {
  SingleThreadedResultHandler(WriteOutputHandler);
}

void SingleStreamWriter::SingleThreadedResultHandler(
    void (*Handler)(
      HostOutVertexContent*,
      HostOutEdgeContent*,
      ostream *out)) {
  static HostOutVertexContent out_vcon;
  static HostOutEdgeContent out_econ;

  static bool host_data_initialized = false;

  static unsigned int vcon_copied_size = 0;
  static unsigned int econ_copied_size = 0;
  static AutoMutex copy_mutex;

  copy_mutex.Lock();
  if (!host_data_initialized) {
    out_vcon.Allocate(global->d_num_vertex);
    out_econ.Allocate(global->d_num_edge);
    host_data_initialized = true;
  }

  if (vcon->d_size > 0) {
    out_vcon.CopyFromDevice(*vcon, 0, vcon->d_size);
    vcon_copied_size += vcon->d_size;
  }
  if (econ->d_size > 0) {
    out_econ.CopyFromDevice(*econ, 0, econ->d_size);
    econ_copied_size += econ->d_size;
  }

  if (vcon_copied_size == global->d_num_vertex &&
      econ_copied_size == global->d_num_edge) {
    (*Handler)(&out_vcon, &out_econ, out);

    out_vcon.Deallocate();
    out_econ.Deallocate();

    host_data_initialized = false;
    vcon_copied_size = 0;
    econ_copied_size = 0;
  }
  copy_mutex.Unlock();
}

ostream* SingleStreamWriter::out = NULL;
