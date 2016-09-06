// *****************************************************************************
// Filename:    file_reader.cc
// Date:        2012-12-10 19:09
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "file_reader.h"

#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

#include "generated_io_data_types.h"
#include "shared_data.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "------> [FileReader " << reader_id << "]: "
#endif

using std::cerr;
using std::cout;
using std::stringstream;
using std::endl;
using std::ifstream;
using std::string;

FileReader::FileReader(const Config *conf, const unsigned int in_reader_id)
    : GraphReader(conf, in_reader_id) {
  string filename;
  if (conf->GetNumReadingThreads() == 1) {
    filename = conf->GetInputFile();
  } else {
    stringstream ss;
    ss << std::setw(2) << std::setfill('0') << reader_id;
    ss << "-of-";
    ss << std::setw(2) << std::setfill('0') << conf->GetNumReadingThreads();
    ss >> filename;
  }
  cout << "FileReader NO. " << reader_id
       << ", opening file: " << filename << endl;
  in.open(filename.c_str());
}

void FileReader::ReadGlobal(IoGlobal *global) {
  IoGlobal::Read(in, global);
}

void FileReader::ReadVertexContent(SharedData *shared_data) {
  // Data to read:
  // num_vertex_for_current_shard
  // vid_1 in_edge_count_1 out_edge_count_1 user_defined_member_a_1 ...
  // ...
  // vid_n in_edge_count_n out_edge_count_n user_defined_member_a_n ...

  unsigned int num_vertex_for_current_shard = 0;
  in >> num_vertex_for_current_shard;
  IoVertex v;
  for (unsigned int i = 0; i < num_vertex_for_current_shard; ++i) {
    IoVertex::Read(in, &v);
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

void FileReader::ReadEdgeContent(SharedData *shared_data) {
  // Data to read:
  // num_edge_for_current_shard
  // from_1 to_1 user_defined_member_a_1 user_defined_member_b_1 ...
  // ...
  // from_m to_m user_defined_member_a_m user_defined_member_b_m ...

  unsigned int num_edge_for_current_shard = 0;
  in >> num_edge_for_current_shard;
  IoEdge e;
  for (unsigned int i = 0; i < num_edge_for_current_shard; ++i) {
    IoEdge::Read(in, &e);
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
}
