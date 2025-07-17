#pragma once
#include "tensor.h"
class Sampler {
 public:
  virtual int32_t sample(const Tensor &cls) = 0;
};

class GreedySampler : public Sampler {
 public:
  int32_t sample(const Tensor &cls) override;
};

class Top_P_Sampler : public Sampler {
 public:
  Top_P_Sampler(float temp, float top_p);
  int32_t sample(const Tensor &cls) override;

 private:
  float m_temp;
  float m_top_p;
};

class SamplerDispatcher : public Sampler {
 public:
  SamplerDispatcher(float temp = -1, float top_p = 0.0f)
      : greedy_sampler(std::make_unique<GreedySampler>()),
        top_p_sampler(std::make_unique<Top_P_Sampler>(temp, top_p)),
        m_temp(temp) {}

  int32_t sample(const Tensor &cls) override {
    if (m_temp <= 0.0f) {
      return greedy_sampler->sample(cls);
    } else {
      return top_p_sampler->sample(cls);
    }
  }

 private:
  std::unique_ptr<Sampler> greedy_sampler;
  std::unique_ptr<Sampler> top_p_sampler;
  float m_temp;
};
