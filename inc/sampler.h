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