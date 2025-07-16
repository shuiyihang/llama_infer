#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "base.h"
#include "tensor.h"

class Layer {
 public:
  static constexpr int MAX_INPUT_OUTPUT = 4;
  explicit Layer(LayerType type, std::string name = "");

  virtual Status forward();

  Status forward(const Tensor &input1, const Tensor &output1);
  Status forward(const Tensor &input1, const Tensor &input2, const Tensor &output1);
  Status forward(const Tensor &input1, const Tensor &input2, const Tensor &input3, const Tensor &output1);
  Status forward(const Tensor &input1, const Tensor &input2, const Tensor &input3, const Tensor &input4,
                 const Tensor &output1);

 protected:
  void set_input(int idx, const Tensor &input);
  void set_output(int idx, const Tensor &output);
  Tensor &get_input(int idx = 0);
  Tensor &get_output(int idx = 0);

 protected:
  LayerType m_layer_type;
  std::string m_layer_name;
  std::vector<Tensor> m_input;
  std::vector<Tensor> m_output;
};

class ParamLayer : public Layer {
 public:
  ParamLayer(LayerType type, std::string name = "");
  void set_weight(int32_t idx, const Tensor &tensor);
  size_t set_weight(int32_t idx, const std::vector<int32_t> &dims, const void *data, DataType type);
  const Tensor &get_weight(int32_t idx = 0) const;

 protected:
  size_t calc_elem_nums(const std::vector<int32_t> &dims);
  std::vector<Tensor> m_weights;
};

// 将tokenid ==> n*dim向量
class EmbeddingLayer : public ParamLayer {
 public:
  explicit EmbeddingLayer();
  Status forward() override;
};

class RmsNormLayer : public ParamLayer {
 public:
  RmsNormLayer(std::string name);
  Status forward() override;
};

class MatMulLayer : public ParamLayer {
 public:
  MatMulLayer(std::string name, bool has_bias = false);
  size_t set_bias(int32_t dim, const void *bias_data, DataType type);
  Status forward() override;

 private:
  bool m_has_bias;
  Tensor m_bias;
};

class RoPELayer : public ParamLayer {
 public:
  explicit RoPELayer(std::string name);
  size_t set_fcos_cache(const std::vector<int32_t> &dims, const void *data, DataType type);
  size_t set_fsin_cache(const std::vector<int32_t> &dims, const void *data, DataType type);
  Status forward() override;

 private:
  Tensor m_fcos;
  Tensor m_fsin;
};

class VecAddLayer : public Layer {
 public:
  inline VecAddLayer() : Layer(LayerType::kLayerAdd, "add") {
    m_input.resize(2);
    m_output.resize(1);
  }
  Status forward() override;
};

class SwiGLULayer : public Layer {
 public:
  inline SwiGLULayer() : Layer(LayerType::kLayerSwiGLU, "swiglu") {
    m_input.resize(2);  // @gate @up
    m_output.resize(1);
  }
  Status forward() override;
};

class MultiHeadAttentionLayer : public Layer {
 public:
  explicit MultiHeadAttentionLayer(int32_t kv_head_num, int32_t head_num, int32_t head_size, const Tensor &kcache,
                                   const Tensor &vcache, const Tensor &score);
  Status forward() override;
  void set_params(int32_t layer, int32_t pos);

 private:
  int32_t m_layer;
  int32_t m_pos;
  int32_t m_mem_num;
  int32_t m_head_num;
  int32_t m_head_size;
  Tensor m_k_cache;
  Tensor m_v_cache;
  Tensor m_score;
};