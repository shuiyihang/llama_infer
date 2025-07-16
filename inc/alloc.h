#pragma once
#include <cstdlib>
#include <memory>
#include "base.h"

class MemAllocator {
 public:
  virtual void *allocate(size_t size) = 0;
  virtual void release(void *ptr) = 0;

 private:
  DeviceType m_device_type = DeviceType::kDeviceUnknown;
};

class CPUMemAllocator : public MemAllocator {
 public:
  void *allocate(size_t size) override;
  void release(void *ptr) override;

  static MemAllocator *instance() {
    static std::unique_ptr<CPUMemAllocator> allocator(new CPUMemAllocator);
    return allocator.get();
  }
  ~CPUMemAllocator() = default;

 private:
  CPUMemAllocator() = default;

  CPUMemAllocator(const CPUMemAllocator &) = delete;
  CPUMemAllocator &operator=(const CPUMemAllocator &) = delete;
};