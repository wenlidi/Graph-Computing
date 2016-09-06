// *****************************************************************************
// Filename:    dummy_writer.h
// Date:        2013-01-01 17:42
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef DUMMY_WRITER_H_
#define DUMMY_WRITER_H_

#include "output_writer.h"

class DummyWriter : public OutputWriter {
 public:

  DummyWriter() : OutputWriter(NULL, NULL, NULL) {
  }

  virtual void WriteOutput() {
  }
};

#endif
