// *****************************************************************************
// Filename:    message_content_manager.h
// Date:        2013-01-07 11:10
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef MESSAGE_CONTENT_MANAGER_H_
#define MESSAGE_CONTENT_MANAGER_H_

struct MessageContent;

namespace thrust {

template <class T>
class device_ptr;

}  // namespace

class MessageContentManager {
 public:

  static void Allocate(const unsigned int size, MessageContent *mcon);

  static void Deallocate(MessageContent *mcon);

  static void Shuffle(
      MessageContent *mcon,
      thrust::device_ptr<unsigned int> thr_shuffle_index,
      void *d_tmp_buf);

  static void Copy(const MessageContent &from, MessageContent *to);

#ifndef LAMBDA_FULL_MESSAGE_IN_EACH_SUPERSTEP
  static void Clear(MessageContent *mcon);
#endif

#ifdef LAMBDA_DEBUG
  static void DebugOutput(const MessageContent &mcon, const bool is_send_buf);
#endif
};

#endif
