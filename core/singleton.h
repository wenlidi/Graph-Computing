// *****************************************************************************
// Filename:    singleton.h
// Date:        2012-12-26 15:26
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef SINGLETON_H_
#define SINGLETON_H_

#include "auto_mutex.h"

template <class T>
class Singleton {
 public:

  static T* GetInstance() {
    if (instance == NULL) {
      mutex.Lock();
      if (instance == NULL) {
        instance = CreateInstance();
      }
      mutex.Unlock();
    }
    return instance;
  }

 private:

  static AutoMutex mutex;

  static T *instance;

  // Let user implement this function.
  static T* CreateInstance();

  // No implementation.
  Singleton();
  Singleton(const Singleton &);
  Singleton& operator=(const Singleton &);

  class SingletonDestructor {
    ~SingletonDestructor() {
      if (Singleton::instance != NULL) {
        delete Singleton::instance;
      }
    }
  };
  static SingletonDestructor destructor;
};

template <class T>
AutoMutex Singleton<T>::mutex;

template <class T>
T* Singleton<T>::instance = NULL;

#endif
