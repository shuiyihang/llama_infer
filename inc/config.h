#pragma once
#include <cstdint>

struct ModelConfig {
  int32_t dim = 0;
  int32_t hidden_dim = 0;
  int32_t layer_num = 0;
  int32_t head_num = 0;
  int32_t kv_head_num = 0;
  int32_t vocab_size = 0;
  int32_t seq_len = 0;  // 上下文最大长度
};

struct TransformerConfig {
  int32_t m_dim;
  int32_t m_hidden_dim;   // ffn中上采样到的维度
  int32_t m_layer_num;    // blk 块数
  int32_t m_q_head_num;   // 多头注意力中的头数
  int32_t m_kv_head_num;  // GQA中注意力头分为 m_kv_head_num 组
  int32_t m_ctx_len;      // 上下文最大token数
  int32_t m_head_size;    // 多头注意力查询实现中：head_size = dim / q_head_num
  int32_t m_kv_dim;  // GQA中同一组共享kv, 实现中: kv_dim由所有组的head_size拼接 —— kv_dim = head_size * kv_head_num
  int32_t m_mem_num;  // 同一组中的查询个数 mem_num = q_head_num / kv_head_num
  int32_t m_vocab_size;
  int32_t freq_cache_size;
  bool m_shared_token_weight;
};
