#include "buffer.h"
#include <cstring>

Buffer::Buffer(size_t size, MemAllocator *allocator, void *ptr)
    : m_size(size), m_ptr(ptr), m_allocator(allocator), m_owner(false) {
  if (!ptr && allocator) {
    m_owner = true;
    m_ptr = m_allocator->allocate(m_size);
  }
}

Buffer::Buffer(const Buffer &other) {
  // 拷贝构造,尽可能深拷贝
  // 如果buffer使用的是外部的，说明不易释放，应该可以共享
  if (other.m_owner && other.m_allocator) {
    m_owner = true;
    m_ptr = m_allocator->allocate(m_size);
    m_size = other.m_size;
    memcpy(m_ptr, other.m_ptr, m_size);
    m_allocator = other.m_allocator;
  } else {
    m_owner = false;
    m_ptr = other.m_ptr;
    m_size = other.m_size;
    m_allocator = other.m_allocator;
  }
}

Buffer::~Buffer() {
  if (m_owner && m_ptr) {
    m_allocator->release(m_ptr);
  }
}

void *Buffer::ptr() { return m_ptr; }
const void *Buffer::ptr() const { return m_ptr; }