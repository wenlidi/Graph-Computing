// *****************************************************************************
// Filename:    global_manager.h
// Date:        2013-01-07 11:33
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GLOBAL_MANAGER_H_
#define GLOBAL_MANAGER_H_

struct IoGlobal;
struct Global;

class GlobalManager {
 public:

  static void Set(const IoGlobal &src, Global *dst);

#ifdef LAMBDA_DEBUG
  static void DebugOutput(const Global &global);
#endif
};

#endif
