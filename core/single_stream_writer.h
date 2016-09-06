// *****************************************************************************
// Filename:    single_stream_writer.h
// Date:        2012-12-26 14:04
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef SINGLE_STREAM_WRITER_H_
#define SINGLE_STREAM_WRITER_H_

#include <ostream>

#include "device_graph_data_types.h"
#include "host_out_graph_data_types.h"
#include "output_writer.h"
#include "auto_mutex.h"

using std::ostream;

class SingleStreamWriter : public OutputWriter {
 public:

  // TODO(laigd): Refractor the code to underline that 'Global' in each thread
  // should be the same.
  SingleStreamWriter(Global *g, VertexContent *v, EdgeContent *e);

  virtual void WriteOutput();

 protected:

  void SetOStream(ostream *o) {
    out = o;
  }

  void SingleThreadedResultHandler(
      void (*Handler)(
          HostOutVertexContent*,
          HostOutEdgeContent*,
          ostream *out));

 private:

  static ostream *out;
};

#endif
