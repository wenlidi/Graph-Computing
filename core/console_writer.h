// *****************************************************************************
// Filename:    single_console_writer.h
// Date:        2012-12-26 14:30
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef SINGLE_CONSOLE_WRITER_H_
#define SINGLE_CONSOLE_WRITER_H_

#include <iostream>

#include "single_stream_writer.h"

using std::cout;

struct Global;
struct VertexContent;
struct EdgeContent;

class ConsoleWriter : public SingleStreamWriter {
 public:

  ConsoleWriter(Global *g, VertexContent *v, EdgeContent *e)
      : SingleStreamWriter(g, v, e) {
    SetOStream(&cout);
  }

};

#endif
