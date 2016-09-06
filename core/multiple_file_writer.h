// *****************************************************************************
// Filename:    multiple_file_writer.h
// Date:        2012-12-26 20:55
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef MULTIPLE_FILE_WRITER_H_
#define MULTIPLE_FILE_WRITER_H_

#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

#include "config.h"
#include "host_out_graph_data_types.h"
#include "output_writer.h"

using std::ofstream;
using std::stringstream;
using std::string;

struct Global;
struct VertexContent;
struct EdgeContent;

class MultipleFileWriter : public OutputWriter {
 public:

  MultipleFileWriter(
      const Config *conf,
      const unsigned int writer_id,
      Global *g,
      VertexContent *v,
      EdgeContent *e);

  virtual void WriteOutput();

 private:

  static const unsigned int kOutBufSize = 256;

  ofstream out;

  HostOutVertexContent out_vcon;

  HostOutEdgeContent out_econ;

};

#endif
