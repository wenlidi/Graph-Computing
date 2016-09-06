// *****************************************************************************
// Filename:    file_test_writer.cc
// Date:        2013-01-04 17:12
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "file_test_writer.h"

#include <fstream>

#include "auto_mutex.h"

using std::ofstream;

ofstream FileTestWriter::out;

AutoMutex FileTestWriter::out_mutex;
