// *****************************************************************************
// Filename:    util.cc
// Date:        2012-12-11 14:20
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#include "util.h"

#include <cmath>
#include <sstream>
#include <string>

using std::stringstream;
using std::string;

unsigned int Util::GetCountForPartX(
    const unsigned int total,
    const unsigned int num_part,
    const unsigned int part_x_id) {
  if (part_x_id >= num_part) return 0;
  return total / num_part + (part_x_id < total % num_part ? 1 : 0);
}

unsigned int Util::GetAccumulatedCountBeforePartX(
    const unsigned int total,
    const unsigned int num_part,
    const unsigned int part_x_id) {
  if (part_x_id >= num_part) return total;
  if (part_x_id == 0) return 0;

  unsigned int q = total / num_part;
  unsigned int r = total % num_part;
  if (r == 0) {
    return part_x_id * q;
  } else {
    unsigned int num_larger_parts = std::min(part_x_id, r);
    return (q + 1) * num_larger_parts + q * (part_x_id - num_larger_parts);
  }
}

string Util::IToA(const int val) {
  stringstream ss;
  string out;
  ss << val;
  ss >> out;
  return out;
}
