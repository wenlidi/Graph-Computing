// *****************************************************************************
// Filename:    test_writer.cc
// Date:        2012-12-31 15:41
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "test_writer.h"

#include <ostream>
#include <cmath>

#include <gflags/gflags.h>
#include "host_graph_data_types.h"
#include "host_out_graph_data_types.h"
#include "shared_data.h"
#include "device_graph_data_types.h"
#include "single_stream_writer.h"

DEFINE_bool(write_test_result, false, "");

using std::ostream;
using std::endl;

namespace {

#include "result_compare.h"

const Global *g = NULL;
SharedData *sd = NULL;

void TestResultHandler(
    HostOutVertexContent *gpu_vout,
    HostOutEdgeContent *gpu_eout,
    ostream *out) {
  vector<HostGraphVertex> *cpu_vout = NULL;
  vector<HostGraphEdge> *cpu_eout = NULL;
  sd->RunCPUSingleThreadAlgorithm(&cpu_vout, &cpu_eout);
  bool vertex_result_correct = false;
  bool edge_result_correct = false;
  CompareResult(
      gpu_vout, gpu_eout, cpu_vout, cpu_eout,
      &vertex_result_correct, &edge_result_correct);

  if (vertex_result_correct) {
    (*out) << "Vertex result correct." << endl;
  } else {
    (*out) << "Vertex result doesn't match!!" << endl;
  }

  if (edge_result_correct) {
    (*out) << "Edge result correct." << endl;
  } else {
    (*out) << "Edge result doesn't match!!" << endl;
  }

  if (FLAGS_write_test_result) {
    if (!vertex_result_correct || !edge_result_correct) {
      HostOutGlobal::Write(*g, *out);

      (*out) << "Vertex result from device (GPU):" << endl;
      gpu_vout->Write(*out);
      (*out) << "Vertex result from CPU:" << endl;
      for (unsigned int i = 0; i < cpu_vout->size(); ++i) {
        cpu_vout->at(i).Write(*out);
      }

      (*out) << "Edge result from device (GPU):" << endl;
      gpu_eout->Write(*out);
      (*out) << "Edge result from CPU:" << endl;
      for (unsigned int i = 0; i < cpu_eout->size(); ++i) {
        cpu_eout->at(i).Write(*out);
      }
    }
  }
}

}  // namespace

void TestWriter::WriteOutput() {
  sd = shared_data;
  g = global;
  SingleThreadedResultHandler(TestResultHandler);
}
