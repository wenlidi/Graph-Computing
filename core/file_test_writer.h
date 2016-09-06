// *****************************************************************************
// Filename:    file_test_writer.h
// Date:        2013-01-04 17:09
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef FILE_TEST_WRITER_H_
#define FILE_TEST_WRITER_H_

#include <fstream>

#include "config.h"
#include "test_writer.h"

using std::ofstream;

struct Global;
struct VertexContent;
struct EdgeContent;
class SharedData;

class FileTestWriter : public TestWriter {
 public:

  FileTestWriter(
      const Config *conf,
      Global *g,
      VertexContent *v,
      EdgeContent *e,
      SharedData *s)
      : TestWriter(g, v, e, s) {
    if (!out.is_open()) {
      out_mutex.Lock();
      if (!out.is_open()) {
        out.open(conf->GetOutputFile().c_str());
        SetOStream(&out);
      }
      out_mutex.Unlock();
    }
  }

 private:

  static ofstream out;

  static AutoMutex out_mutex;
};

#endif
