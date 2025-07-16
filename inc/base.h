#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

enum class DeviceType : uint8_t { kDeviceUnknown = 0, KDeviceCpu = 1, KDeviceGPU = 2 };

enum class DataType : uint8_t { kDataTypeUnknown = 0, kDataTypeFp32, kDataTypeInt32 };

inline size_t DataTypeSize(DataType type) {
  switch (type) {
    case DataType::kDataTypeFp32:
      return sizeof(float);
    case DataType::kDataTypeInt32:
      return sizeof(int32_t);
    default:
      break;
  }
  return 0;
}

enum class LayerType : uint8_t {
  kLayerUnknown = 0,
  kLayerLinear,
  kLayerEncode,     // 将string => token id
  kLayerEmbedding,  // token id => 1*dim向量
  kLayerRMSNorm,
  kLayerMatmul,
  kLayerRoPE,
  kLayerSoftmax,
  kLayerMHA,
  kLayerAdd,
  kLayerSwiGLU,
};

enum class StatusCode : uint8_t { kSuccess = 0, kFailed };

class Status {
 public:
  Status(StatusCode code = StatusCode::kSuccess, std::string msg = "") : m_code(code), m_msg(std::move(msg)) {}

  const std::string &get_err_msg() const;

  bool operator==(const Status &other) { return m_code == other.m_code; }
  operator bool() const { return m_code == StatusCode::kSuccess; }

 private:
  StatusCode m_code;
  std::string m_msg;
};

enum class ModelBufferType : uint8_t {
  kBufferTypeUnknown = 0,
  kBufferEmbeddingInput,
  kBufferFcosCache,
  kBufferFsinCache,
  kBufferRMSNorm,
  kBufferMHA,
  kBufferGate,
  kBufferUp,
  kBufferDown,
  kBufferKCache,
  kBufferVCache,
  kBufferQuery,
  kBufferScore,
  kBufferCls,
  kBufferPos,
  kBufferTokenId,
  kBufferAttnOutPut,
};

enum class TokenizerType : uint8_t {
  kVocabTypeUnknown = 0,
  kVocabTypeBpe,
  kVocabTypeSpe,
};