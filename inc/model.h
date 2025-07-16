#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include "base.h"
#include "config.h"
#include "encode.h"
#include "sampler.h"
#include "tensor.h"

struct RawModelData {
  RawModelData() : m_fd(-1), m_file_size(0), m_data(nullptr), m_weight(nullptr) {}
  ~RawModelData();

  int32_t m_fd;
  size_t m_file_size;
  void *m_data;
  void *m_weight;

  virtual const void *weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData {
  const void *weight(size_t offset) const override;
};

class Model {
 public:
  explicit Model(TokenizerType vocab_type, std::string ckpt_pth, std::string tokenizer_pth);

  virtual void init() = 0;
  virtual int32_t forward(const Tensor &input, int32_t pos) = 0;

 protected:
  virtual Status load_model_from_file();
  virtual Status insert_dict(ModelBufferType key, Tensor &value);
  virtual Tensor &get_tensor(ModelBufferType key);
  virtual void create_layers() = 0;

 private:
  virtual void generate_model_info(const ModelConfig &config);

 protected:
  TokenizerType m_vocab_type;
  std::unique_ptr<EncodeLayerBase> m_encode_layer;
  std::string m_ckpt_pth;
  std::string m_tokenizer_pth;
  std::unordered_map<ModelBufferType, Tensor> m_dict;

  std::unique_ptr<RawModelData> m_raw_data;
  std::unique_ptr<TransformerConfig> m_config;

  std::unique_ptr<Sampler> m_sampler;
};