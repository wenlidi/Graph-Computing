// *****************************************************************************
// Filename:    adjustable-heap.h
// Date:        2012-10-27 16:58
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef ADJUSTABLE_HEAP_H_
#define ADJUSTABLE_HEAP_H_

#include <algorithm>
#include <vector>

using std::vector;

#define ADD_ADJUSTABLE_HEAP_MEMBER() \
    unsigned int heap_position

#define PARENT(I) (((I) - 1) >> 1)
#define LEFT(I) (((I) << 1) + 1)
#define RIGHT(I) (((I) + 1) << 1)

#define OFFSETTED_PARENT(I) ((I) >> 1)
#define OFFSETTED_LEFT(I) ((I) << 1)
#define OFFSETTED_RIGHT(I) (((I) << 1) + 1)

#define LESS(A, B) (*(A) < *(B))
#define HEAPPOS(A) ((A)->heap_position)

static const unsigned int kInvalidPos = ~0U;

// This class template is a heap containing pointers to a set of objects. In
// addition to general minimum-heap, it supports adjusting the position of the
// ojbects dynamically after user modifing the value in the object which may
// change the order. The adjusting is done by invoking UpHeap and DownHeap.
// What's more, user can get each object's position in the heap at any time via
// the object's data member @heap_position.
//
// NOTE: Those objects could not be deleted while user using this class to
// perform heap-related operations on them.
template<typename T>
class AdjustableHeap {
 public:
  explicit AdjustableHeap(const unsigned int reserved_size = 256) : ptr_heap() {
    ptr_heap.reserve(reserved_size);
  }

  // require that *iterator is T.
  template<typename ForwardIterator>
  AdjustableHeap(ForwardIterator first,
                 ForwardIterator last,
                 const unsigned int reserved_size = 256) {
    ptr_heap.reserve(reserved_size);
    for (unsigned int i = 0; first != last; ++i) {
      ptr_heap.push_back(&(*first));  // should not just use push_back(first).
      HEAPPOS(ptr_heap.back()) = i;
      ++first;
    }
    MakeHeap();
  }

  ~AdjustableHeap() {
  }

  unsigned int Size() {
    return ptr_heap.size();
  }

  // We use const because we don't want the heap_position member to be modified
  // even if they don't intend to do so.
  const T* Top() const {
    return ptr_heap[0];
  }

  T* Pop() {
    T *&heap0 = ptr_heap[0];
    T *top = heap0;
    HEAPPOS(top) = kInvalidPos;

    heap0 = ptr_heap.back();
    ptr_heap.pop_back();
    HEAPPOS(heap0) = 0;

    DownHeap(0);
    return top;
  }

  // Pushes an object pointer into the heap, and returns the final position of
  // it.
  unsigned int Push(T *element) {
    HEAPPOS(element) = ptr_heap.size();
    ptr_heap.push_back(element);
    return UpHeap(ptr_heap.size() - 1);
  }

  // Let the object pointer residing in position @pos perform an 'up heap'
  // operation, and returns the final position.
  unsigned int UpHeap(unsigned int pos) {
    while (pos > 0) {
      unsigned int p = PARENT(pos);
      if (LESS(ptr_heap[pos], ptr_heap[p])) {
        T *&lhs = ptr_heap[pos], *&rhs = ptr_heap[p];
        std::swap(lhs, rhs);
        std::swap(HEAPPOS(lhs), HEAPPOS(rhs));
        pos = p;
      } else {
        break;
      }
    }
    return pos;
  }

  // Let the object pointer residing in position @pos perform an 'down heap'
  // operation, and returns the final position.
  unsigned int DownHeap(unsigned int pos) {
    while (true) {
      unsigned int l = LEFT(pos);
      unsigned int r = RIGHT(pos);
      unsigned int smallest = pos;
      if (l < ptr_heap.size() && LESS(ptr_heap[l], ptr_heap[smallest])) {
        smallest = l;
      }
      if (r < ptr_heap.size() && LESS(ptr_heap[r], ptr_heap[smallest])) {
        smallest = r;
      }
      if (smallest == pos) break;

      // Reference to a pointer.
      T *&lhs = ptr_heap[smallest], *&rhs = ptr_heap[pos];
      std::swap(lhs, rhs);
      std::swap(HEAPPOS(lhs), HEAPPOS(rhs));

      pos = smallest;
    }
    return pos;
  }

  void SortHeap(vector<T*> *out) {
    out->clear();
    out->reserve(ptr_heap.size());
    while (ptr_heap.size() > 0) {
      out->push_back(ptr_heap.Pop());
    }
  }

 private:
  vector<T*> ptr_heap;

  void MakeHeap() {
    for (unsigned int i = ptr_heap.size() >> 1; i > 0; --i) {
      DownHeap(i);
    }
    DownHeap(0);
  }
};

#undef PARENT
#undef LEFT
#undef RIGHT

#undef OFFSETTED_PARENT
#undef OFFSETTED_LEFT
#undef OFFSETTED_RIGHT

#undef LESS
#undef HEAPPOS

#endif
