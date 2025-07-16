#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include "layer.h"
#include "model.h"
#include "tensor.h"

struct Qwen2Layers {
  // 有参数层
  std::unique_ptr<ParamLayer> m_embedding;
  std::unique_ptr<ParamLayer> m_cls;

  // blk[0..n]
  // self attn
  std::vector<std::unique_ptr<ParamLayer>> m_input_layernorm;
  std::vector<std::unique_ptr<ParamLayer>> m_q_proj;
  std::vector<std::unique_ptr<ParamLayer>> m_k_proj;
  std::vector<std::unique_ptr<ParamLayer>> m_v_proj;
  std::vector<std::unique_ptr<ParamLayer>> m_o_proj;

  // mlp
  std::vector<std::unique_ptr<ParamLayer>> m_post_layernorm;
  std::vector<std::unique_ptr<ParamLayer>> m_up;
  std::vector<std::unique_ptr<ParamLayer>> m_gate;
  std::vector<std::unique_ptr<ParamLayer>> m_down;

  std::unique_ptr<ParamLayer> m_rope;
  std::unique_ptr<ParamLayer> m_final_layernorm;

  // 无参数层, 公用
  std::unique_ptr<Layer> m_add;
  std::unique_ptr<Layer> m_swiglu;
  std::unique_ptr<Layer> m_mha;
};

class Qwen2Model : public Model {
 public:
  explicit Qwen2Model(std::string ckpt_pth, std::string tokenizer_pth);
  void init() override;

  std::vector<int32_t> encode(std::string &prompt);
  std::string decode(std::vector<int32_t> &tokens);
  Tensor fill_input(int32_t token);
  // 输出预测的tokenid
  int32_t forward(const Tensor &input, int32_t pos) override;
  bool is_sentence_ending(int32_t next);

 private:
  void create_layers() override;
  void init_mem();
  void create_param_layers();
  void create_nonparam_layers();

  void input_rmsnorm_blk(int32_t layer, const Tensor &input);
  void calc_qkv_blk(int32_t layer, int32_t pos);
  void calc_mha_blk(int32_t layer, int32_t pos);
  void mlp_blk(int32_t layer, const Tensor &input);
  void cls_logits(const Tensor &input);

  std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer, int32_t pos);

 private:
  std::unique_ptr<Qwen2Layers> m_layers;
};
