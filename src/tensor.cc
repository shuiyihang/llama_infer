#include "tensor.h"
#include <cstddef>
#include <cstdint>
#include <utility>

Tensor::Tensor(DataType data_type, std::vector<int32_t> dims, MemAllocator *allocator, void *ptr)
    : m_data_type(data_type), m_dims(std::move(dims)) {
  m_elem_nums = elem_nums();  // must first
  m_buffer = std::make_shared<Buffer>(byte_size(), allocator, ptr);
}

void Tensor::assign(std::unique_ptr<Buffer> buffer) { m_buffer = std::move(buffer); }

template <typename T>
T *Tensor::ptr(size_t offset) {
  if (!m_buffer) return nullptr;
  return reinterpret_cast<T *>(m_buffer->ptr()) + offset;
}

template <typename T>
const T *Tensor::ptr(size_t offset) const {
  if (!m_buffer) return nullptr;
  return reinterpret_cast<T *>(m_buffer->ptr()) + offset;
}

size_t Tensor::byte_size() const { return size() * DataTypeSize(m_data_type); }

size_t Tensor::size() const { return m_elem_nums; }

size_t Tensor::elem_nums() const {
  size_t res = 1;
  for (auto it = m_dims.begin(); it != m_dims.end(); it++) {
    res *= *it;
  }
  return res;
}

const std::vector<int32_t> &Tensor::shape() const { return m_dims; }

Tensor::Tensor(Tensor &&other) noexcept {
  this->m_data_type = other.m_data_type;
  this->m_dims = std::move(other.m_dims);
  this->m_buffer = std::move(other.m_buffer);
  this->m_elem_nums = other.m_elem_nums;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  this->m_data_type = other.m_data_type;
  this->m_dims = std::move(other.m_dims);
  this->m_buffer = std::move(other.m_buffer);
  this->m_elem_nums = other.m_elem_nums;
  return *this;
}

// 显式实例化
template float *Tensor::ptr(size_t offset);
template const float *Tensor::ptr(size_t offset) const;
template int32_t *Tensor::ptr(size_t offset);
template const int32_t *Tensor::ptr(size_t offset) const;