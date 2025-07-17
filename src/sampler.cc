#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

int32_t GreedySampler::sample(const Tensor &cls) {
  const float *logits = cls.ptr<float>();
  size_t next = 0;
  for (size_t i = 1; i < cls.size(); i++) {
    if (logits[i] > logits[next]) {
      next = i;
    }
  }
  return next;
}

Top_P_Sampler::Top_P_Sampler(float temp, float top_p) : m_temp(temp), m_top_p(top_p) {}
int32_t Top_P_Sampler::sample(const Tensor &cls) {
  const float *logits = cls.ptr<float>();
  size_t len = cls.size();
  float max_logit = *std::max_element(logits, logits + len);

  using token_prob = std::pair<int32_t, float>;
  std::vector<token_prob> probs;  // (token_id, prob)
  for (size_t i = 0; i < len; i++) {
    probs.emplace_back(i, std::exp((logits[i] - max_logit) / m_temp));
  }

  // 降序排列prob
  std::sort(probs.begin(), probs.end(), [](const token_prob &a, const token_prob &b) {
    return a.second > b.second;  // sort 返回true，表示第一个参数应该排在第二个参数前
  });

  float cumsum = 0.0f;
  std::vector<token_prob> top_p_set;
  for (const auto &p : probs) {
    cumsum += p.second;
    top_p_set.push_back(p);
    if (cumsum >= m_top_p) break;
  }

  // norm
  float top_p_sum = 0.0f;
  for (const auto &p : top_p_set) {
    top_p_sum += p.second;
  }
  for (auto &p : top_p_set) {
    p.second /= top_p_sum;
  }

  // gen
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  float r = dis(gen);
  float acc = 0.0f;
  for (const auto &p : top_p_set) {
    acc += p.second;
    if (acc >= r) return p.first;
  }

  return top_p_set.back().first;
}