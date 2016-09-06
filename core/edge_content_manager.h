// *****************************************************************************
// Filename:    edge_content_manager.h
// Date:        2013-01-07 11:01
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef EDGE_CONTENT_MANAGER_H_
#define EDGE_CONTENT_MANAGER_H_

struct EdgeContent;

namespace thrust {

template <class T>
class device_ptr;

}  // namespace

class EdgeContentManager {
 public:

  static void Allocate(const unsigned int size, EdgeContent *econ);

  static void Deallocate(EdgeContent *econ);

  static void ShuffleInMembers(
      EdgeContent *econ,
      thrust::device_ptr<unsigned int> &thr_shuffle_index,
      void *d_tmp_buf);

  static void InitOutMembers(EdgeContent *econ);

#ifdef LAMBDA_DEBUG
  static void DebugOutput(const EdgeContent &econ);
#endif
};

#endif
