#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include "layer.h"
#include "set"

namespace tiktoken {
class tiktoken;
}

class EncodeLayerBase : public Layer {
 public:
  explicit EncodeLayerBase(std::string tokenizer_pth)
      : Layer(LayerType::kLayerEncode, "Encode"), m_tokenizer_pth(std::move(tokenizer_pth)) {}

  virtual std::vector<int32_t> encode(const std::string &sentence) const = 0;
  virtual std::string decode(std::vector<int32_t> &token) const = 0;
  virtual int32_t vocab_size() const = 0;
  virtual bool is_sentence_ending(int32_t token) = 0;

 protected:
  std::string m_tokenizer_pth;
};

class BpeEncodeLayer : public EncodeLayerBase {
 public:
  explicit BpeEncodeLayer(std::string tokenizer_pth);
  std::vector<int32_t> encode(const std::string &sentence) const override;
  std::string decode(std::vector<int32_t> &token) const override;
  int32_t vocab_size() const override;
  bool is_sentence_ending(int32_t token) override;

 protected:
  std::set<int32_t> m_eog_tokens;
  int32_t m_num_tokens;
  std::unique_ptr<tiktoken::tiktoken> m_tiktoken;
};