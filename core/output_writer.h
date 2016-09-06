// *****************************************************************************
// Filename:    output_writer.h
// Date:        2012-12-26 12:05
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef OUTPUT_WRITER_H_
#define OUTPUT_WRITER_H_

#include "device_graph_data_types.h"

class OutputWriter {
 public:

  OutputWriter(Global *g, VertexContent *v, EdgeContent *e)
      : global(g),
        vcon(v),
        econ(e) {
  }

  virtual ~OutputWriter() {
  }

  virtual void WriteOutput() = 0;

 protected:

  Global *global;
  VertexContent *vcon;
  EdgeContent *econ;
};

#endif
