#include "alloc.h"

void *CPUMemAllocator::allocate(size_t size) {
  if (!size) {
    return nullptr;
  }
  return malloc(size);
}
void CPUMemAllocator::release(void *ptr) {
  if (ptr) {
    free(ptr);
  }
}