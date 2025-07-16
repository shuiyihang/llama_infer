#include "sampler.h"
#include <cstddef>

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