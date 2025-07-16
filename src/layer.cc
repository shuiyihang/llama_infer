#include "layer.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include "base.h"
#include "buffer.h"
#include "op.h"
#include "tensor.h"

Layer::Layer(LayerType type, std::string name)
    : m_layer_type(type), m_layer_name(std::move(name)), m_input(MAX_INPUT_OUTPUT), m_output(MAX_INPUT_OUTPUT) {}

void Layer::set_input(int idx, const Tensor &input) {
  if (idx >= MAX_INPUT_OUTPUT) {
    fprintf(stderr, "idx out of range\n");
    return;
  }
  m_input.at(idx) = input;
}
void Layer::set_output(int idx, const Tensor &output) {
  if (idx >= MAX_INPUT_OUTPUT) {
    fprintf(stderr, "idx out of range\n");
    return;
  }
  m_output.at(idx) = output;
}

Tensor &Layer::get_input(int idx) { return m_input.at(idx); }
Tensor &Layer::get_output(int idx) { return m_output.at(idx); }

Status Layer::forward(const Tensor &input1, const Tensor &output1) {
  set_input(0, input1);
  set_output(0, output1);
  return forward();
}
Status Layer::forward(const Tensor &input1, const Tensor &input2, const Tensor &output1) {
  set_input(0, input1);
  set_input(1, input2);
  set_output(0, output1);
  return forward();
}
Status Layer::forward(const Tensor &input1, const Tensor &input2, const Tensor &input3, const Tensor &output1) {
  set_input(0, input1);
  set_input(1, input2);
  set_input(2, input3);
  set_output(0, output1);
  return forward();
}
Status Layer::forward(const Tensor &input1, const Tensor &input2, const Tensor &input3, const Tensor &input4,
                      const Tensor &output1) {
  set_input(0, input1);
  set_input(1, input2);
  set_input(2, input3);
  set_input(3, input4);
  set_output(0, output1);
  return forward();
}

Status Layer::forward() { return Status(); }

ParamLayer::ParamLayer(LayerType type, std::string name) : Layer(type, std::move(name)) {}

size_t ParamLayer::calc_elem_nums(const std::vector<int32_t> &dims) {
  size_t res = 1;
  for (auto it = dims.begin(); it != dims.end(); it++) {
    res *= *it;
  }
  return res;
}

size_t ParamLayer::set_weight(int32_t idx, const std::vector<int32_t> &dims, const void *data, DataType type) {
  size_t size = calc_elem_nums(dims);
  Tensor weight(type, dims);
  std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(weight.byte_size(), nullptr, const_cast<void *>(data));
  weight.assign(std::move(buffer));
  m_weights.at(idx) = std::move(weight);

  return size;
}

const Tensor &ParamLayer::get_weight(int32_t idx) const { return m_weights.at(idx); }

EmbeddingLayer::EmbeddingLayer() : ParamLayer(LayerType::kLayerEmbedding, "embedding") {
  m_weights.resize(1);
  m_input.resize(1);  // 输入tokenid
  m_output.resize(1);
}
Status EmbeddingLayer::forward() {
  CPU_OP::embedding_op(get_weight(), get_input(), get_output());
  return Status();
}

RmsNormLayer::RmsNormLayer(std::string name) : ParamLayer(LayerType::kLayerRMSNorm, std::move(name)) {
  m_weights.resize(1);
  m_input.resize(1);
  m_output.resize(1);
}
Status RmsNormLayer::forward() {
  // weight: 读权重文件时已经保存
  CPU_OP::rmsnorm_op(get_weight(), get_input(), get_output());
  return Status();
}

MatMulLayer::MatMulLayer(std::string name, bool has_bias)
    : ParamLayer(LayerType::kLayerMatmul, std::move(name)), m_has_bias(has_bias) {
  m_weights.resize(1);
  m_input.resize(1);
  m_output.resize(1);
}

Status MatMulLayer::forward() {
  CPU_OP::matmul_op(get_weight(), get_input(), get_output());

  if (m_has_bias) {
    CPU_OP::matadd_op(get_output(), m_bias, get_output());
  }
  return Status();
}

size_t MatMulLayer::set_bias(int32_t dim, const void *bias_data, DataType type) {
  Tensor bias(type, {dim});
  std::unique_ptr<Buffer> buffer = std::make_unique<Buffer>(bias.byte_size(), nullptr, const_cast<void *>(bias_data));
  bias.assign(std::move(buffer));
  m_bias = std::move(bias);

  return dim;
}

RoPELayer::RoPELayer(std::string name) : ParamLayer(LayerType::kLayerRoPE, std::move(name)) {
  m_input.resize(3);  // q,k,pos
  m_output.resize(1);
}

size_t RoPELayer::set_fcos_cache(const std::vector<int32_t> &dims, const void *data, DataType type) {
  Tensor fcos(type, dims);
  auto buffer = std::make_unique<Buffer>(fcos.byte_size(), nullptr, const_cast<void *>(data));
  fcos.assign(std::move(buffer));
  m_fcos = std::move(fcos);
  return calc_elem_nums(dims);
}
size_t RoPELayer::set_fsin_cache(const std::vector<int32_t> &dims, const void *data, DataType type) {
  Tensor fsin(type, dims);
  auto buffer = std::make_unique<Buffer>(fsin.byte_size(), nullptr, const_cast<void *>(data));
  fsin.assign(std::move(buffer));
  m_fsin = std::move(fsin);
  return calc_elem_nums(dims);
}

Status RoPELayer::forward() {
  CPU_OP::rope_op(get_input(0), get_input(1), get_input(2), m_fsin, m_fcos);
  return Status();
}

MultiHeadAttentionLayer::MultiHeadAttentionLayer(int32_t mem_num, int32_t head_num, int32_t head_size,
                                                 const Tensor &kcache, const Tensor &vcache, const Tensor &score)
    : Layer(LayerType::kLayerMHA, "mha"),
      m_k_cache(kcache),
      m_v_cache(vcache),
      m_score(score),
      m_mem_num(mem_num),
      m_head_num(head_num),
      m_head_size(head_size) {
  m_input.resize(1);  // q
  m_output.resize(1);
}

void MultiHeadAttentionLayer::set_params(int32_t layer, int32_t pos) {
  m_layer = layer;
  m_pos = pos;
}

Status MultiHeadAttentionLayer::forward() {
  CPU_OP::mha_op(m_layer, m_pos, m_mem_num, m_head_num, m_head_size, get_input(), m_k_cache, m_v_cache, m_score,
                 get_output());
  return Status();
}

Status VecAddLayer::forward() {
  CPU_OP::matadd_op(get_input(0), get_input(1), get_output());
  return Status();
}

Status SwiGLULayer::forward() {
  CPU_OP::swiglu_op(get_input(0), get_input(1), get_output());
  return Status();
}