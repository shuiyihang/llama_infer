#include "qwen2.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include "base.h"
#include "layer.h"
#include "tensor.h"

Qwen2Model::Qwen2Model(std::string ckpt_pth, std::string tokenizer_pth)
    : Model(TokenizerType::kVocabTypeBpe, std::move(ckpt_pth), std::move(tokenizer_pth)) {
  m_layers = std::make_unique<Qwen2Layers>();
}

void Qwen2Model::init() {
  load_model_from_file();
  m_sampler = std::make_unique<GreedySampler>();
}

std::vector<int32_t> Qwen2Model::encode(std::string &prompt) { return m_encode_layer->encode(prompt); }
std::string Qwen2Model::decode(std::vector<int32_t> &tokens) { return m_encode_layer->decode(tokens); }

Tensor Qwen2Model::fill_input(int32_t token) {
  auto &embedding_input = get_tensor(ModelBufferType::kBufferEmbeddingInput);
  auto &input_token = get_tensor(ModelBufferType::kBufferTokenId);
  *input_token.ptr<int32_t>() = token;
  m_layers->m_embedding->forward(input_token, embedding_input);
  return embedding_input;
}

void Qwen2Model::input_rmsnorm_blk(int32_t layer, const Tensor &input) {
  auto &rms_output = get_tensor(ModelBufferType::kBufferRMSNorm);
  m_layers->m_input_layernorm.at(layer)->forward(input, rms_output);
}
/*
1: ==> Q,K,V
2: ==> Q,K--rope--> Q,K
*/
void Qwen2Model::calc_qkv_blk(int32_t layer, int32_t pos) {
  auto &query = get_tensor(ModelBufferType::kBufferQuery);
  auto [key, val] = slice_kv_cache(layer, pos);

  auto &rms_output = get_tensor(ModelBufferType::kBufferRMSNorm);

  // rms_input@wq ==> Q
  m_layers->m_q_proj.at(layer)->forward(rms_output, query);

  m_layers->m_k_proj.at(layer)->forward(rms_output, key);
  m_layers->m_v_proj.at(layer)->forward(rms_output, val);

  auto &t_pos = get_tensor(ModelBufferType::kBufferPos);
  *t_pos.ptr<int32_t>() = pos;

  m_layers->m_rope->forward(query, key, t_pos, Tensor());
}

void Qwen2Model::calc_mha_blk(int32_t layer, int32_t pos) {
  auto &query = get_tensor(ModelBufferType::kBufferQuery);
  auto &mha_output = get_tensor(ModelBufferType::kBufferMHA);
  // 含有虚函数的类转换
  dynamic_cast<MultiHeadAttentionLayer *>(m_layers->m_mha.get())->set_params(layer, pos);
  m_layers->m_mha->forward(query, mha_output);

  // 还要经过一个线性层 @wo
  auto &attn_output = get_tensor(ModelBufferType::kBufferAttnOutPut);
  m_layers->m_o_proj.at(layer)->forward(mha_output, attn_output);
}

/**
input 为 token映射后的向量
*/
void Qwen2Model::mlp_blk(int32_t layer, const Tensor &input) {
  // 进入mlp之前：
  // 1. 残差连接
  // 2. rmsnorm
  m_layers->m_add->forward(input, get_tensor(ModelBufferType::kBufferAttnOutPut), input);

  auto &ffn_rmsnorm = get_tensor(ModelBufferType::kBufferRMSNorm);
  m_layers->m_post_layernorm.at(layer)->forward(input, ffn_rmsnorm);

  auto &gate_output = get_tensor(ModelBufferType::kBufferGate);
  m_layers->m_gate.at(layer)->forward(ffn_rmsnorm, gate_output);

  auto &up_output = get_tensor(ModelBufferType::kBufferUp);
  m_layers->m_up.at(layer)->forward(ffn_rmsnorm, up_output);

  m_layers->m_swiglu->forward(gate_output, up_output, gate_output);

  auto &down_output = get_tensor(ModelBufferType::kBufferDown);
  m_layers->m_down.at(layer)->forward(gate_output, down_output);

  // 再进行一次残差连接
  m_layers->m_add->forward(down_output, input, input);
}
void Qwen2Model::cls_logits(const Tensor &input) {
  // 1. rmsnorm
  // 2. cls 线性层
  m_layers->m_final_layernorm->forward(input, input);
  auto &cls_output = get_tensor(ModelBufferType::kBufferCls);
  m_layers->m_cls->forward(input, cls_output);
}

int32_t Qwen2Model::forward(const Tensor &input, int32_t pos) {
  int32_t next;
  for (int i = 0; i < m_config->m_layer_num; i++) {
    input_rmsnorm_blk(i, input);
    calc_qkv_blk(i, pos);
    calc_mha_blk(i, pos);
    mlp_blk(i, input);
  }
  cls_logits(input);
  next = m_sampler->sample(get_tensor(ModelBufferType::kBufferCls));
  return next;
}

bool Qwen2Model::is_sentence_ending(int32_t next) { return m_encode_layer->is_sentence_ending(next); }

std::pair<Tensor, Tensor> Qwen2Model::slice_kv_cache(int32_t layer, int32_t pos) {
  size_t offset = layer * m_config->m_ctx_len * m_config->m_kv_dim + pos * m_config->m_kv_dim;
  float *k_cache = get_tensor(ModelBufferType::kBufferKCache).ptr<float>(offset);
  float *v_cache = get_tensor(ModelBufferType::kBufferVCache).ptr<float>(offset);

  Tensor k(DataType::kDataTypeFp32, {m_config->m_kv_dim}, nullptr, k_cache);
  Tensor v(DataType::kDataTypeFp32, {m_config->m_kv_dim}, nullptr, v_cache);

  return std::pair<Tensor, Tensor>{std::move(k), std::move(v)};
}

void Qwen2Model::create_param_layers() {
  // 从模型中加载参数
  size_t offset = 0;
  m_layers->m_embedding = std::make_unique<EmbeddingLayer>();
  const void *weight_data = m_raw_data->weight(offset);
  offset += m_layers->m_embedding->set_weight(0, {m_config->m_vocab_size, m_config->m_dim}, weight_data,
                                              DataType::kDataTypeFp32);

  // input_layernorm.weight
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto rmsnorm = std::make_unique<RmsNormLayer>("input_rmsnorm_" + std::to_string(i));
    weight_data = m_raw_data->weight(offset);
    offset += rmsnorm->set_weight(0, {m_config->m_dim}, weight_data, DataType::kDataTypeFp32);
    m_layers->m_input_layernorm.emplace_back(std::move(rmsnorm));
  }
  // self_attn.q_proj
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto q_proj = std::make_unique<MatMulLayer>("q_proj" + std::to_string(i), true);
    offset +=
        q_proj->set_weight(0, {m_config->m_dim, m_config->m_dim}, m_raw_data->weight(offset), DataType::kDataTypeFp32);

    offset += q_proj->set_bias(m_config->m_dim, m_raw_data->weight(offset), DataType::kDataTypeFp32);
    m_layers->m_q_proj.emplace_back(std::move(q_proj));
  }
  // self_attn.k_proj
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto k_proj = std::make_unique<MatMulLayer>("k_proj" + std::to_string(i), true);
    offset += k_proj->set_weight(0, {m_config->m_kv_dim, m_config->m_dim}, m_raw_data->weight(offset),
                                 DataType::kDataTypeFp32);

    offset += k_proj->set_bias(m_config->m_kv_dim, m_raw_data->weight(offset), DataType::kDataTypeFp32);
    m_layers->m_k_proj.emplace_back(std::move(k_proj));
  }
  // self_attn.v_proj
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto v_proj = std::make_unique<MatMulLayer>("v_proj" + std::to_string(i), true);
    offset += v_proj->set_weight(0, {m_config->m_kv_dim, m_config->m_dim}, m_raw_data->weight(offset),
                                 DataType::kDataTypeFp32);

    offset += v_proj->set_bias(m_config->m_kv_dim, m_raw_data->weight(offset), DataType::kDataTypeFp32);
    m_layers->m_v_proj.emplace_back(std::move(v_proj));
  }
  // self_attn.o_proj.weight
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto o_proj = std::make_unique<MatMulLayer>("o_proj" + std::to_string(i), false);  // no bias
    offset +=
        o_proj->set_weight(0, {m_config->m_dim, m_config->m_dim}, m_raw_data->weight(offset), DataType::kDataTypeFp32);
    m_layers->m_o_proj.emplace_back(std::move(o_proj));
  }
  // post_attention_layernorm.weight
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto rmsnorm = std::make_unique<RmsNormLayer>("post_rmsnorm_" + std::to_string(i));
    weight_data = m_raw_data->weight(offset);
    offset += rmsnorm->set_weight(0, {m_config->m_dim}, weight_data, DataType::kDataTypeFp32);
    m_layers->m_post_layernorm.emplace_back(std::move(rmsnorm));
  }
  // mlp
  // gate_proj.weight
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto gate = std::make_unique<MatMulLayer>("gate_" + std::to_string(i), false);  // no bias
    offset += gate->set_weight(0, {m_config->m_hidden_dim, m_config->m_dim}, m_raw_data->weight(offset),
                               DataType::kDataTypeFp32);
    m_layers->m_gate.emplace_back(std::move(gate));
  }
  // down_proj.weight
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto down = std::make_unique<MatMulLayer>("down_" + std::to_string(i), false);  // no bias
    offset += down->set_weight(0, {m_config->m_dim, m_config->m_hidden_dim}, m_raw_data->weight(offset),
                               DataType::kDataTypeFp32);
    m_layers->m_down.emplace_back(std::move(down));
  }
  // up_proj.weight
  for (int i = 0; i < m_config->m_layer_num; i++) {
    auto up = std::make_unique<MatMulLayer>("up_" + std::to_string(i), false);  // no bias
    offset += up->set_weight(0, {m_config->m_hidden_dim, m_config->m_dim}, m_raw_data->weight(offset),
                             DataType::kDataTypeFp32);
    m_layers->m_up.emplace_back(std::move(up));
  }
  // model.norm.weight
  auto final_rmsnorm = std::make_unique<RmsNormLayer>("final_rmsnorm");
  weight_data = m_raw_data->weight(offset);
  offset += final_rmsnorm->set_weight(0, {m_config->m_dim}, weight_data, DataType::kDataTypeFp32);
  m_layers->m_final_layernorm = std::move(final_rmsnorm);

  // sin cos cache
  auto rope = std::make_unique<RoPELayer>("RoPE");  // 1,048,576
  weight_data = m_raw_data->weight(offset);         // 494032768
  offset +=
      rope->set_fcos_cache({m_config->m_ctx_len, m_config->freq_cache_size}, weight_data, DataType::kDataTypeFp32);
  weight_data = m_raw_data->weight(offset);  // 495081344
  offset +=
      rope->set_fsin_cache({m_config->m_ctx_len, m_config->freq_cache_size}, weight_data, DataType::kDataTypeFp32);
  m_layers->m_rope = std::move(rope);

  if (m_raw_data->weight(offset) != static_cast<char *>(m_raw_data->m_data) + m_raw_data->m_file_size) {
    fprintf(stderr, "file parase failed!\n");
    exit(-1);
  } else {
    fprintf(stdout, "param read success!\n");
  }
  // output cls weight
  m_layers->m_cls = std::make_unique<MatMulLayer>("cls", false);
  if (m_config->m_shared_token_weight) {
    weight_data = m_raw_data->weight(0);
  } else {
    weight_data = m_raw_data->weight(offset);
  }
  m_layers->m_cls->set_weight(0, {m_config->m_vocab_size, m_config->m_dim}, weight_data, DataType::kDataTypeFp32);
}

void Qwen2Model::create_nonparam_layers() {
  m_layers->m_swiglu = std::make_unique<SwiGLULayer>();
  m_layers->m_add = std::make_unique<VecAddLayer>();
  m_layers->m_mha = std::make_unique<MultiHeadAttentionLayer>(
      m_config->m_mem_num, m_config->m_q_head_num, m_config->m_head_size, get_tensor(ModelBufferType::kBufferKCache),
      get_tensor(ModelBufferType::kBufferVCache), get_tensor(ModelBufferType::kBufferScore));
}

void Qwen2Model::init_mem() {
  auto allocator = CPUMemAllocator::instance();

  Tensor input_pos(DataType::kDataTypeInt32, {1}, allocator);
  Tensor input_token(DataType::kDataTypeInt32, {1}, allocator);
  // 存储token映射为向量，残差连接的源
  Tensor embedding_input(DataType::kDataTypeFp32, {m_config->m_dim}, allocator);
  Tensor fsin_cache(DataType::kDataTypeFp32, {m_config->m_ctx_len, m_config->freq_cache_size}, allocator);
  Tensor fcos_cache(DataType::kDataTypeFp32, {m_config->m_ctx_len, m_config->freq_cache_size}, allocator);
  Tensor rms_output(DataType::kDataTypeFp32, {m_config->m_dim}, allocator);
  Tensor gate_output(DataType::kDataTypeFp32, {m_config->m_hidden_dim}, allocator);
  Tensor up_output(DataType::kDataTypeFp32, {m_config->m_hidden_dim}, allocator);
  Tensor kcache(DataType::kDataTypeFp32, {m_config->m_layer_num, m_config->m_ctx_len, m_config->m_kv_dim}, allocator);
  Tensor vcache(DataType::kDataTypeFp32, {m_config->m_layer_num, m_config->m_ctx_len, m_config->m_kv_dim}, allocator);
  // 映射后向量经rms,Q*之后
  Tensor query(DataType::kDataTypeFp32, {m_config->m_dim}, allocator);

  // TODO 后面添加注释
  Tensor score(DataType::kDataTypeFp32, {m_config->m_q_head_num, m_config->m_ctx_len}, allocator);
  Tensor cls(DataType::kDataTypeFp32, {m_config->m_vocab_size}, allocator);

  insert_dict(ModelBufferType::kBufferPos, input_pos);
  insert_dict(ModelBufferType::kBufferTokenId, input_token);
  insert_dict(ModelBufferType::kBufferEmbeddingInput, embedding_input);
  insert_dict(ModelBufferType::kBufferFsinCache, fsin_cache);
  insert_dict(ModelBufferType::kBufferFcosCache, fcos_cache);
  // 共用
  insert_dict(ModelBufferType::kBufferRMSNorm, rms_output);
  insert_dict(ModelBufferType::kBufferMHA, rms_output);
  insert_dict(ModelBufferType::kBufferDown, rms_output);

  insert_dict(ModelBufferType::kBufferGate, gate_output);
  insert_dict(ModelBufferType::kBufferUp, up_output);
  insert_dict(ModelBufferType::kBufferKCache, kcache);
  insert_dict(ModelBufferType::kBufferVCache, vcache);
  // 共用
  insert_dict(ModelBufferType::kBufferQuery, query);
  insert_dict(ModelBufferType::kBufferAttnOutPut, query);

  insert_dict(ModelBufferType::kBufferScore, score);
  insert_dict(ModelBufferType::kBufferCls, cls);
}

void Qwen2Model::create_layers() {
  create_param_layers();
  init_mem();
  create_nonparam_layers();
}