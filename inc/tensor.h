
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "base.h"
#include "buffer.h"

class Tensor {
 public:
  explicit Tensor() = default;
  explicit Tensor(DataType data_type, std::vector<int32_t> dims, MemAllocator *allocator = nullptr,
                  void *ptr = nullptr);

  void assign(std::unique_ptr<Buffer> buffer);

  size_t byte_size() const;
  size_t size() const;

  const std::vector<int32_t> &shape() const;

  template <typename T>
  T *ptr(size_t offset = 0);

  template <typename T>
  const T *ptr(size_t offset = 0) const;

  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  Tensor(const Tensor &other) = default;
  Tensor &operator=(const Tensor &other) = default;

 private:
  inline size_t elem_nums() const;

 private:
  DataType m_data_type;
  std::vector<int32_t> m_dims;
  std::shared_ptr<Buffer> m_buffer;
  size_t m_elem_nums;
};