// *****************************************************************************
// Filename:    single_file_writer.h
// Date:        2012-12-26 10:42
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef SINGLE_FILE_WRITER_H_
#define SINGLE_FILE_WRITER_H_

#include <fstream>

#include "auto_mutex.h"
#include "config.h"
#include "single_stream_writer.h"

using std::ofstream;

struct Global;
struct VertexContent;
struct EdgeContent;

class SingleFileWriter : public SingleStreamWriter {
 public:

  SingleFileWriter(
      const Config *conf,
      Global *g,
      VertexContent *v,
      EdgeContent *e) : SingleStreamWriter(g, v, e) {
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
