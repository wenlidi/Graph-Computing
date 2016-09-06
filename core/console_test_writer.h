// *****************************************************************************
// Filename:    console_test_writer.h
// Date:        2013-01-04 17:04
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef CONSOLE_TEST_WRITER_H_
#define CONSOLE_TEST_WRITER_H_

#include <iostream>

#include "test_writer.h"

using std::cout;

struct Global;
struct VertexContent;
struct EdgeContent;
class SharedData;

class ConsoleTestWriter : public TestWriter {
 public:

  ConsoleTestWriter(Global *g, VertexContent *v, EdgeContent *e, SharedData *s)
      : TestWriter(g, v, e, s) {
    SetOStream(&cout);
  }
};

#endif
