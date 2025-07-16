#pragma once
#include <cstddef>
#include "alloc.h"
class Buffer {
 public:
  explicit Buffer(size_t size, MemAllocator *allocator = nullptr, void *ptr = nullptr);
  ~Buffer();

  Buffer(const Buffer &other);

  void *ptr();
  const void *ptr() const;

 private:
  bool m_owner;
  size_t m_size;  // 字节数
  void *m_ptr;
  MemAllocator *m_allocator;
};