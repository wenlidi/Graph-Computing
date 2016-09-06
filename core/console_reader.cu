// *****************************************************************************
// Filename:    console_reader.cu
// Date:        2013-03-25 15:24
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "console_reader.h"

#include <iostream>
#include <cstdlib>

#include "generated_io_data_types.h"
#include "shared_data.h"

using std::cin;
using std::cerr;

ConsoleReader::ConsoleReader(const Config *conf, const unsigned int in_reader_id)
    : GraphReader(conf, in_reader_id) {
  if (conf->GetNumReadingThreads() != 1) {
    cout << "Error: number of reading threads should be 1." << endl;
    exit(1);
  }
}

void ConsoleReader::ReadGlobal(IoGlobal *global) {
  IoGlobal::Read(cin, global);
}

void ConsoleReader::ReadVertexContent(SharedData *shared_data) {
  // Data to read:
  // num_vertex_for_current_shard
  // vid_1 in_edge_count_1 out_edge_count_1 user_defined_member_a_1 ...
  // ...
  // vid_n in_edge_count_n out_edge_count_n user_defined_member_a_n ...

  unsigned int num_vertex_for_current_shard = 0;
  cin >> num_vertex_for_current_shard;
  IoVertex v;
  for (unsigned int i = 0; i < num_vertex_for_current_shard; ++i) {
    IoVertex::Read(cin, &v);
    shared_data->AddVertex(v);
  }
}

void ConsoleReader::ReadEdgeContent(SharedData *shared_data) {
  // Data to read:
  // num_edge_for_current_shard
  // from_1 to_1 user_defined_member_a_1 user_defined_member_b_1 ...
  // ...
  // from_m to_m user_defined_member_a_m user_defined_member_b_m ...

  unsigned int num_edge_for_current_shard = 0;
  cin >> num_edge_for_current_shard;
  IoEdge e;
  for (unsigned int i = 0; i < num_edge_for_current_shard; ++i) {
    IoEdge::Read(cin, &e);
    shared_data->AddEdge(e);
  }
}
