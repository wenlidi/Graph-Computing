// *****************************************************************************
// Filename:    reading_thread_data_types.cc
// Date:        2012-12-09 22:33
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "reading_thread_data_types.h"

#include <algorithm>
#include <iostream>
#include <string>

#include "generated_io_data_types.h"
#include "graph_reader.h"
#include "console_reader.h"
#include "file_reader.h"
#include "simple_reader.h"
#include "rand_reader.h"
#include "rmat_reader.h"
#include "util.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

#ifdef LAMBDA_DEBUG
#include "debug.h"
#define LAMBDA_HEADER "---> [ReadingThread " << reading_thread_id << "]: "
#endif

#ifdef LAMBDA_PROFILING
#include "singleton.h"
#include "profiler.h"
#endif

ReadingThreadData::ReadingThreadData()
    : reading_thread_id(~0U),
      conf(NULL),
      shared_data(NULL),
      reader(NULL) {
}

ReadingThreadData::~ReadingThreadData() {
  if (reader != NULL) delete reader;
}

void ReadingThreadData::Init(
    const unsigned int id,
    const Config *config,
    SharedData *shared) {
  reading_thread_id = id;
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "ReadingThreadData::Init..." << endl;
  );
#endif
  conf = config;
  shared_data = shared;

  if (conf->GetGraphType() == GraphType::kGraphFromConsole) {
    reader = new ConsoleReader(conf, reading_thread_id);
  } else if (conf->GetGraphType() == GraphType::kGraphFromFile) {
    reader = new FileReader(conf, reading_thread_id);
  } else if (conf->GetGraphType() == GraphType::kSimpleGraph) {
    reader = new SimpleReader(conf, reading_thread_id);
  } else if (conf->GetGraphType() == GraphType::kRMatGraph) {
    reader = new RMatReader(conf, reading_thread_id);
  } else if (conf->GetGraphType() == GraphType::kRandGraph) {
    reader = new RandReader(conf, reading_thread_id);
  }
}

void ReadingThreadData::Run() {
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "ReadingThreadData::Run()..." << endl;
  );
#endif
  if (reading_thread_id == 0) {
    // Read global and copy to each gpu control thread
    IoGlobal g;
    reader->ReadGlobal(&g);
    shared_data->SetGlobalOnce(g);
#ifdef LAMBDA_DEBUG
    DBG_WRAP_COUT(
    cout << LAMBDA_HEADER << "Finish reading Global." << endl;
    );
#endif
    // Signal gpu control threads that global is ready
    shared_data->SignalGlobalRead();
  } else {
    shared_data->WaitForGlobalRead();
#ifdef LAMBDA_DEBUG
    DBG_WRAP_COUT(
    cout << LAMBDA_HEADER << "Finish waiting for Global read." << endl;
    );
#endif
  }

  // Wait for vcon memory ready.
  shared_data->WaitForVconMemoryReady();
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "Finish waiting for vcon memory." << endl;
  );
#endif

#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StartTimer(
      "Read vcon " + Util::IToA(reading_thread_id));
#endif
  // Read vcon and signal gpu control threads when finished.
  reader->ReadVertexContent(shared_data);
  shared_data->FlushVconBuffer();
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StopTimer(
      "Read vcon " + Util::IToA(reading_thread_id));
#endif
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "Finish reading & copying vcon." << endl;
  );
#endif

  // Wait for econ memory ready
  shared_data->WaitForEconMemoryReady();
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "Finish waiting for econ memory." << endl;
  );
#endif

#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StartTimer(
      "Read econ " + Util::IToA(reading_thread_id));
#endif
  // Read econ and signal gpu control threads when finished.
  reader->ReadEdgeContent(shared_data);
  shared_data->FlushEconBuffer();
#ifdef LAMBDA_PROFILING
  Singleton<Profiler>::GetInstance()->StopTimer(
      "Read econ " + Util::IToA(reading_thread_id));
#endif
#ifdef LAMBDA_DEBUG
  DBG_WRAP_COUT(
  cout << LAMBDA_HEADER << "Finish reading & copying econ." << endl;
  );
#endif
}

#ifdef LAMBDA_DEBUG
#undef LAMBDA_HEADER
#endif
