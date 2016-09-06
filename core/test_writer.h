// *****************************************************************************
// Filename:    test_writer.h
// Date:        2012-12-31 15:38
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef TEST_WRITER_H_
#define TEST_WRITER_H_

#include <iostream>

#include <gflags/gflags.h>
#include "device_graph_data_types.h"
#include "shared_data.h"
#include "single_stream_writer.h"

DECLARE_bool(write_test_result);

using std::cout;

class TestWriter : public SingleStreamWriter {
 public:

  virtual void WriteOutput();

 protected:

  TestWriter(Global *g, VertexContent *v, EdgeContent *e, SharedData *s)
      : SingleStreamWriter(g, v, e), shared_data(s) {
  }

 private:

  SharedData *shared_data;
};

#endif
