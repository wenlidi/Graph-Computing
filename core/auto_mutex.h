// *****************************************************************************
// Filename:    auto_mutex.h
// Date:        2012-12-26 15:24
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef AUTO_MUTEX_H_
#define AUTO_MUTEX_H_

#include <pthread.h>

class AutoMutex {
 public:

  AutoMutex() {
    pthread_mutex_init(&auto_mutex, 0);
  }

  ~AutoMutex() {
    pthread_mutex_destroy(&auto_mutex);
  }

  void Lock() {
    pthread_mutex_lock(&auto_mutex);
  }

  void Unlock() {
    pthread_mutex_unlock(&auto_mutex);
  }

 private:

  pthread_mutex_t auto_mutex;
};

#endif
