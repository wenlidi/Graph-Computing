// *****************************************************************************
// Filename:    profiler.cc
// Date:        2013-01-03 18:13
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "profiler.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <string>
#include <vector>
#include <list>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

using std::cout;
using std::endl;
using std::map;
using std::string;
using std::vector;
using std::list;

void Profiler::Clear() {
  timer_map.clear();
  timer_list.clear();
}

void Profiler::StartTimer(const string &name) {
  pthread_mutex_lock(&map_mutex);

  map<string, TimerValue*>::iterator it = timer_map.find(name);
  TimerValue *ptr = NULL;

  if (it == timer_map.end()) {
    // Create a new timer.
    timer_list.resize(timer_list.size() + 1);
    ptr = &timer_list.back();
    timer_map[name] = ptr;
    sdkCreateTimer(&(ptr->timer));
  } else {
    if (it->second->is_profiling) {
      cout << "Profiler::StartTimer error: timer "
           << name << " already started!" << endl;
      exit(1);
    }
    ptr = it->second;
  }

  ptr->is_profiling = true;
  sdkResetTimer(&(ptr->timer));
  sdkStartTimer(&(ptr->timer));

  pthread_mutex_unlock(&map_mutex);
}

void Profiler::StopTimer(const string &name) {
  pthread_mutex_lock(&map_mutex);

  map<string, TimerValue*>::iterator it = timer_map.find(name);
  if (it == timer_map.end()) {
    cout << "Profiler::StopTimer error: could not find timer " << name << endl;
    exit(1);
  }

  TimerValue &val = *(timer_map[name]);
  if (!val.is_profiling) {
    cout << "Profiler::StopTimer error: timer " << name << " is not in use."
         << endl;
    exit(1);
  }
  sdkStopTimer(&(val.timer));
  val.is_profiling = false;
  val.accumulated_duration += sdkGetTimerValue(&(val.timer));

  pthread_mutex_unlock(&map_mutex);
}

void Profiler::Summary() {
  cout << "Profiler summary:" << endl;
  for (map<string, TimerValue*>::iterator it = timer_map.begin();
       it != timer_map.end(); ++it) {
    cout << it->first << ": "
         << std::fixed << std::setprecision(3)
         << std::setfill(' ') << std::setw(9)
         << it->second->accumulated_duration << " ms" << endl;
  }
}

Profiler::Profiler()
  : timer_list(),
    timer_map(),
    map_mutex() {
  pthread_mutex_init(&map_mutex, 0);
}

Profiler::~Profiler() {
  pthread_mutex_destroy(&map_mutex);
}

template <>
Profiler* Singleton<Profiler>::CreateInstance() {
  return new Profiler();
}
