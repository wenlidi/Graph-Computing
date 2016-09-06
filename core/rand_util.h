// *****************************************************************************
// Filename:    rand_util.h
// Date:        2012-12-25 15:43
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef RAND_UTIL_H_
#define RAND_UTIL_H_

#include <cstdlib>
#include <iostream>

using std::endl;
using std::cout;

enum RandType {
  RAND_VERTEX_ID,
  RAND_SMALL_UINT,  // [0, 13)
  RAND_MIDDLE_UINT,  // [0, 65531)
  RAND_LARGE_UINT,  // [0, kMaxUInt)
  RAND_FLOAT,  // [kMinFloat, kMaxFloat]
  RAND_FRACTION,  // [0, 1)
};

class RandUtil {
 public:

  static void SetNumVertexOnce(const unsigned int value) {
    num_vertexes = value;
  }

  static unsigned int RandVertexId() {
    if (num_vertexes == ~0U) {
      cout << "RandUtil::RandVertexId error: number of vertexes not set!"
           << endl;
      exit(1);
    }
    return RandUIntInRange(num_vertexes);
  }

  static unsigned int RandSmallUInt() {
    return RandUIntInRange(13);
  }

  static unsigned int RandMiddleUInt() {
    return RandUIntInRange(65531);
  }

  static unsigned int RandLargeUInt() {
    return rand();
  }

  static float RandFloat() {
    return rand() / 11.0f;
  }

  static float RandFraction() {
    return rand() / (float)RAND_MAX;
  }

 private:

  static unsigned int num_vertexes;

  static unsigned int RandUIntInRange(const unsigned int upper_bound) {
    return rand() % upper_bound;
  }

};

#endif
