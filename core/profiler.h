// *****************************************************************************
// Filename:    profiler.h
// Date:        2013-01-03 18:11
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef PROFILER_H_
#define PROFILER_H_

#include <map>
#include <vector>
#include <list>
#include <string>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "singleton.h"

using std::map;
using std::string;
using std::list;
using std::vector;

class Profiler {
 public:

  // Not multithread-safe.
  void Clear();

  void StartTimer(const string &name);

  void StopTimer(const string &name);

  void Summary();

 private:

  struct TimerValue {
    bool is_profiling;
    StopWatchInterface *timer;
    float accumulated_duration;

    TimerValue()
        : is_profiling(false),
          timer(NULL),
          accumulated_duration(0.0f) {
    }

    ~TimerValue() {
      if (timer != NULL) sdkDeleteTimer(&timer);
    }
  };

  list<TimerValue> timer_list;

  map<string, TimerValue*> timer_map;

  pthread_mutex_t map_mutex;

  Profiler();

  ~Profiler();

  friend class Singleton<Profiler>;
};

#endif
