// *****************************************************************************
// Filename:    single_file_writer.cc
// Date:        2012-12-26 17:11
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "single_file_writer.h"

#include <fstream>

#include "auto_mutex.h"

using std::ofstream;

ofstream SingleFileWriter::out;

AutoMutex SingleFileWriter::out_mutex;
