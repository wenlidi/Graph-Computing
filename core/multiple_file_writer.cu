// *****************************************************************************
// Filename:    multiple_file_writer.cc
// Date:        2013-01-04 16:29
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "multiple_file_writer.h"

#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>

#include "device_graph_data_types.h"

MultipleFileWriter::MultipleFileWriter(
    const Config *conf,
    const unsigned int writer_id,
    Global *g,
    VertexContent *v,
    EdgeContent *e)
    : OutputWriter(g, v, e),
      out(),
      out_vcon(),
      out_econ() {
  stringstream ss;
  ss << std::setw(2) << std::setfill('0')
      << writer_id << "-of-" << conf->GetNumGPUControlThreads();
  string suffix;
  ss >> suffix;
  string filename = conf->GetOutputFile() + "-" + suffix;
  out.open(filename.c_str());

  out_vcon.Allocate(kOutBufSize);
  out_econ.Allocate(kOutBufSize);
}

void MultipleFileWriter::WriteOutput() {
  for (unsigned int copied_size = 0; copied_size < vcon->d_size; ) {
    const unsigned int size_to_copy =
        std::min(kOutBufSize, vcon->d_size - copied_size);
    out_vcon.Clear();
    out_vcon.CopyFromDevice(*vcon, copied_size, size_to_copy);
    out_vcon.Write(out);
  }
  out << endl;

  for (unsigned int copied_size = 0; copied_size < econ->d_size; ) {
    const unsigned int size_to_copy =
        std::min(kOutBufSize, econ->d_size - copied_size);
    out_econ.Clear();
    out_econ.CopyFromDevice(*econ, copied_size, size_to_copy);
    out_econ.Write(out);
  }
}

