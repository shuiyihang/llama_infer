#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include "qwen2.h"
#include "tensor.h"

// 单次最长生成长度
const int32_t MAX_STEPS = 1024;

typedef struct llama_chat_message {
  llama_chat_message(const char *role, const char *content) : m_role(role), m_content(content) {}
  const char *m_role;
  const char *m_content;
} llama_chat_message;

void set_system_prompt(std::vector<llama_chat_message> &msgs) {
  msgs.emplace_back("system",
                    "你是一个本地部署的智能助手，你的名字叫小莫，不是通义千问，也不是其他云端助手。当被问到“你是谁”时，"
                    "请回答:我是小莫，一个本地助手");
}

int32_t llama_chat_apply_template(std::vector<llama_chat_message> &chat, bool add_ass, char *buf, int32_t length) {
  std::stringstream ss;
  for (auto &msg : chat) {
    ss << "<|im_start|>" << msg.m_role << "\n" << msg.m_content << "<|im_end|>\n";
  }
  if (add_ass) {
    ss << "<|im_start|>assistant\n";
  }
  if (buf && length > 0) {
    strncpy(buf, ss.str().data(), length);
  }
  return ss.str().size();
}

int generate(Qwen2Model &model, std::string prompt) {
  static int32_t ctx_pos = 0;
  std::vector<int32_t> tokens = model.encode(prompt);
  int32_t token_len = tokens.size();

  int32_t pos = 0;
  Tensor input;
  int32_t next = tokens[pos];
  bool is_prefill = true;
  while (pos < MAX_STEPS) {
    input = model.fill_input(next);
    next = model.forward(input, ctx_pos + pos);
    if (!is_prefill && model.is_sentence_ending(next)) {
      break;
    }
    pos += 1;
    if (pos < token_len) {
      next = tokens[pos];
    } else {
      // 生成内容
      is_prefill = false;
      std::vector<int32_t> words{next};
      fprintf(stdout, "%s", model.decode(words).data());
      fflush(stdout);
    }
  }
  ctx_pos = ctx_pos + pos + 1;
  return pos;
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "usage: ./chat model.bin tokenizer.json");
  }

  const char *ckpt_pth = argv[1];
  const char *tokenizer_pth = argv[2];
  Qwen2Model model(ckpt_pth, tokenizer_pth);
  model.init();
  fprintf(stdout, "===============新的对话===============\n");
  std::vector<llama_chat_message> msgs;
  std::vector<char> formatted(1024);  // 每一轮最大输入的长度
  set_system_prompt(msgs);
  while (true) {
    printf("\033[32m> \033[0m");
    std::string user;
    std::getline(std::cin, user);
    if (user.empty()) break;
    msgs.emplace_back("user", strdup(user.c_str()));
    int new_len = llama_chat_apply_template(msgs, true, formatted.data(), formatted.capacity());

    std::string_view prompt(formatted.data(), new_len);
    // string_view ==> string只能显式声明
    printf("\033[33m");
    auto start = std::chrono::steady_clock::now();
    int steps = generate(model, std::string(prompt));
    auto end = std::chrono::steady_clock::now();
    printf("\n\033[0m");

    auto duration = std::chrono::duration<double>(end - start).count();
    fprintf(stdout, "(%.3lf tokens/s)\n", steps / duration);
    msgs.clear();
  }

  return 0;
}