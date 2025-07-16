#include "model.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <cstdio>
#include <memory>
#include <utility>
#include "base.h"
#include "config.h"
#include "tiktoken.h"

RawModelData::~RawModelData() {
  if (m_data) {
    munmap(m_data, m_file_size);
  }
  if (m_fd != -1) {
    close(m_fd);
  }
}

const void *RawModelDataFp32::weight(size_t offset) const { return static_cast<float *>(m_weight) + offset; }

Model::Model(TokenizerType vocab_type, std::string ckpt_pth, std::string tokenizer_pth)
    : m_vocab_type(vocab_type), m_ckpt_pth(std::move(ckpt_pth)), m_tokenizer_pth(std::move(tokenizer_pth)) {
  m_encode_layer = std::make_unique<BpeEncodeLayer>(m_tokenizer_pth);
  m_config = std::make_unique<TransformerConfig>();
  m_raw_data = std::make_unique<RawModelDataFp32>();
}

Status Model::load_model_from_file() {
  int fd = open(m_ckpt_pth.data(), O_RDONLY);

  if (fd < 0) {
    return Status(StatusCode::kFailed, "open ckpt failed");
  }
  ModelConfig config;
  read(fd, &config, sizeof(config));

  generate_model_info(config);

  struct stat st;
  fstat(fd, &st);
  m_raw_data->m_file_size = st.st_size;
  m_raw_data->m_fd = fd;
  m_raw_data->m_data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (m_raw_data->m_data == MAP_FAILED) {
    fprintf(stderr, "file mmap failed\n");
  }
  m_raw_data->m_weight = static_cast<char *>(m_raw_data->m_data) + sizeof(ModelConfig);

  create_layers();

  return Status();
}

void Model::generate_model_info(const ModelConfig &config) {
  m_config->m_ctx_len = config.seq_len;
  m_config->m_dim = config.dim;
  m_config->m_hidden_dim = config.hidden_dim;
  m_config->m_layer_num = config.layer_num;
  m_config->m_q_head_num = config.head_num;
  m_config->m_kv_head_num = config.kv_head_num;
  m_config->m_head_size = config.dim / config.head_num;
  m_config->m_kv_dim = m_config->m_head_size * config.kv_head_num;
  m_config->m_mem_num = config.head_num / config.kv_head_num;

  if (config.vocab_size > 0)
    m_config->m_shared_token_weight = true;
  else
    m_config->m_shared_token_weight = false;
  m_config->m_vocab_size = std::abs(config.vocab_size);
  m_config->freq_cache_size = m_config->m_head_size / 2;

  // 左对齐-右对齐
  fprintf(stdout, "%-16s %7d\n", "dim:", config.dim);
  fprintf(stdout, "%-16s %7d\n", "hidden_dim:", config.hidden_dim);
  fprintf(stdout, "%-16s %7d\n", "layer_num:", config.layer_num);
  fprintf(stdout, "%-16s %7d\n", "vocab_size:", config.vocab_size);
  fprintf(stdout, "%-16s %7d\n", "ctx len:", config.seq_len);
  fprintf(stdout, "%-16s %7d\n", "freq cache:", m_config->freq_cache_size);
  fprintf(stdout, "%-16s %7d\n", "GQA head_num:", config.head_num);
  fprintf(stdout, "%-16s %7d\n", "GQA group num:", config.kv_head_num);
  fprintf(stdout, "%-16s %7d\n", "GQA mem num:", m_config->m_mem_num);
}

Status Model::insert_dict(ModelBufferType key, Tensor &value) {
  if (m_dict.count(key) != 0) {
    return Status(StatusCode::kFailed, "repeat insert");
  }
  m_dict.emplace(key, value);
  return Status();
}

Tensor &Model::get_tensor(ModelBufferType key) {
  if (m_dict.count(key) == 0) {
    fprintf(stderr, "key:%d not found\n", static_cast<int>(key));
    exit(-1);
  }
  return m_dict.at(key);
}