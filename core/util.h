// *****************************************************************************
// Filename:    util.h
// Date:        2012-12-11 14:19
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef UTIL_H_
#define UTIL_H_

#include <string>

using std::string;

class Util {
 public:

  static unsigned int GetCountForPartX(
      const unsigned int total,
      const unsigned int num_part,
      const unsigned int part_x_id);

  static unsigned int GetAccumulatedCountBeforePartX(
      const unsigned int total,
      const unsigned int num_part,
      const unsigned int part_x_id);

  static string IToA(const int val);
};

#endif
