// *****************************************************************************
// Filename:    vertex_content_manager.h
// Date:        2013-01-07 10:50
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef VERTEX_CONTENT_MANAGER_H_
#define VERTEX_CONTENT_MANAGER_H_

struct VertexContent;

namespace thrust {

template <class T>
class device_ptr;

}  // namespace

class VertexContentManager {
 public:

  static void Allocate(const unsigned int size, VertexContent *vcon);

  static void Deallocate(VertexContent *vcon);

  static void ShuffleInMembers(
      VertexContent *vcon,
      thrust::device_ptr<unsigned int> &thr_shuffle_index,
      void *d_tmp_buf);

  static void InitOutMembers(VertexContent *vcon);

#ifdef LAMBDA_DEBUG
  static void DebugOutput(const VertexContent &vcon);
#endif
};

#endif
