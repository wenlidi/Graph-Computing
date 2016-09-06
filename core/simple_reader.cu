// *****************************************************************************
// Filename:    simple_reader.cc
// Date:        2012-12-09 23:22
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "simple_reader.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

#include "config.h"
#include "generated_io_data_types.h"
#include "shared_data.h"
#include "util.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> [SimpleReader " << reader_id << "]: "
#endif

SimpleReader::SimpleReader(const Config *conf, const unsigned int in_reader_id)
    : GraphReader(conf, in_reader_id),
      num_vertex_for_current_shard(0),
      num_edge_for_current_shard(0),
      vertex_id_offset(0) {
  Init();
}

void SimpleReader::ReadGlobal(IoGlobal *global) {
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "SimpleReader::ReadGlobal..." << endl;
  );
#endif
  RandGlobal(global);
}

void SimpleReader::ReadVertexContent(SharedData *shared_data) {
  // Data to read:
  // num_vertex_for_current_shard
  // vid_1 in_edge_count_1 out_edge_count_1 user_defined_member_a_1 ...
  // ...
  // vid_n in_edge_count_n out_edge_count_n user_defined_member_a_n ...

  const unsigned int q = num_edge / num_vertex;
  const unsigned int r = num_edge % num_vertex;

  for (int i = 0; i < num_vertex_for_current_shard; ++i) {
    IoVertex v;
    v.id = vertex_id_offset + i;
    if (v.id < r) { // q + 1 edges
      v.out_edge_count = q + 1;
    } else {
      v.out_edge_count = q;
    }
    v.in_edge_count = v.out_edge_count;

    IoVertex::Rand(&v);
#ifdef LAMBDA_DEBUG
    DBG_WRAP_COUT(
    cout << LAMBDA_HEADER
         << "adding vertex: " << v.id << ", "
         << "in_edge_count: " << v.in_edge_count << ", "
         << "out_edge_count: " << v.out_edge_count
         << endl;
    );
#endif
    shared_data->AddVertex(v);
  }
}

void SimpleReader::ReadEdgeContent(SharedData *shared_data) {
  // Data to read:
  // num_edge_for_current_shard
  // from_1 to_1 user_defined_member_a_1 user_defined_member_b_1 ...
  // ...
  // from_m to_m user_defined_member_a_m user_defined_member_b_m ...

  const unsigned int q = num_edge / num_vertex;
  const unsigned int r = num_edge % num_vertex;

  unsigned int total = 0;
  for (int i = 0; i < num_vertex_for_current_shard; ++i) {
    const unsigned int vid = vertex_id_offset + i;
    IoEdge e;
    e.from = vid;

    for (int j = 1; j <= q; ++j) {
      e.to = (vid + j) % num_vertex;
      IoEdge::Rand(&e);
#ifdef LAMBDA_DEBUG
      DBG_WRAP_COUT(
      cout << LAMBDA_HEADER
           << "adding edge from: " << e.from << ", "
           << "to: " << e.to
           << endl;
      );
#endif
      shared_data->AddEdge(e);
    }
    total += q;

    if (vid < r) { // add loop edge
      e.to = vid;
      IoEdge::Rand(&e);
#ifdef LAMBDA_DEBUG
      DBG_WRAP_COUT(
      cout << LAMBDA_HEADER
           << "adding edge from: " << e.from << ", "
           << "to: " << e.to
           << endl;
      );
#endif
      shared_data->AddEdge(e);
      ++total;
    }
  }

  if (total != num_edge_for_current_shard) {
    cout << "Calculation error!" << endl;
    cout << "added edges: " << total << ", "
         << "num_edge_for_current_shard: " << num_edge_for_current_shard
         << endl;
    exit(1);
  }
}

void SimpleReader::Init() {
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "SimpleReader::Init..." << endl;
  );
#endif
  num_vertex_for_current_shard = Util::GetCountForPartX(
      num_vertex, conf->GetNumReadingThreads(), reader_id);

  vertex_id_offset = Util::GetAccumulatedCountBeforePartX(
      num_vertex, conf->GetNumReadingThreads(), reader_id);

  unsigned int accumulated_edges_before = Util::GetAccumulatedCountBeforePartX(
      num_edge, num_vertex, vertex_id_offset);
  unsigned int accumulated_edges_now = Util::GetAccumulatedCountBeforePartX(
      num_edge, num_vertex, vertex_id_offset + num_vertex_for_current_shard);

  num_edge_for_current_shard = accumulated_edges_now - accumulated_edges_before;

#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER
       << "num_vertex_for_current_shard: " << num_vertex_for_current_shard
       << ", "
       << "num_edge_for_current_shard: " << num_edge_for_current_shard << ", "
       << "vertex_id_offset: " << vertex_id_offset
       << endl;
  );
#endif
}

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
