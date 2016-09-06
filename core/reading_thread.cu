// *****************************************************************************
// Filename:    reading_thread.cc
// Date:        2012-12-07 19:20
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "reading_thread.h"

#include <cstdio>
#include <iostream>

#include "reading_thread_data_types.h"

#ifdef LAMBDA_DEBUG
#include "debug.h"
#endif

using std::cout;
using std::endl;

void* ReadingThread(void *args) {
  ReadingThreadData *data = (ReadingThreadData*)args;
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << "ReadingThread starts..." << endl;
  );
#endif
  data->Run();
  return 0;
}
