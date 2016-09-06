// *****************************************************************************
// Filename:    debug.h
// Date:        2012-12-13 15:37
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef DEBUG_H_
#define DEBUG_H_

#include <iostream>
#include <iomanip>

#include "auto_mutex.h"

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

#define DBG_LOCK() cout_lock.Lock()

#define DBG_UNLOCK() cout_lock.Unlock()

#define DBG_WRAP_COUT(CODE) { \
    DBG_LOCK(); \
    CODE; \
    cout.flush(); \
    DBG_UNLOCK(); \
  }

extern AutoMutex cout_lock;

// For debugging device graph data types.
#define DEBUG_OUTPUT_WITH_OFFSET(ORG_BUF, MEMBER, MEMBER_NAME, FROM_OFFSET, COUNT, TYPE) { \
    TYPE* BUF = (TYPE*)ORG_BUF; \
    checkCudaErrors(cudaMemcpy(BUF, MEMBER + FROM_OFFSET, COUNT * sizeof(TYPE), cudaMemcpyDeviceToHost)); \
    cout << LAMBDA_HEADER << "  " << MEMBER_NAME; \
    for (unsigned int i = 0; i < COUNT; ++i) { \
      if (BUF[i] == kMaxUInt) { \
        cout << "  -"; \
      } else { \
        cout << std::setfill(' ') << std::setw(3) << BUF[i]; \
      } \
      cout << ", "; \
    } \
    cout << endl; \
  }

#define DEBUG_OUTPUT(BUF, MEMBER, MEMBER_NAME, COUNT, TYPE) \
    DEBUG_OUTPUT_WITH_OFFSET(BUF, MEMBER, MEMBER_NAME, 0, COUNT, TYPE)

#endif
