#pragma once

#include "tensor.h"

#include "armadillo"

namespace CPU_OP {

void rmsnorm_op(const Tensor &weight, const Tensor &input, Tensor &output);

void matmul_op(const Tensor &weight, const Tensor &input, Tensor &output, float scale = 1.0f);
void matadd_op(const Tensor &input1, const Tensor &input2, Tensor &output);

void rope_op(Tensor &query, Tensor &key, const Tensor &pos, const Tensor &fsin, const Tensor &fcos);

void mha_op(int32_t layer, int32_t pos, int32_t mem_num, int32_t head_num, int32_t head_size, Tensor &query,
            Tensor &k_cache, Tensor &v_cache, Tensor &score, Tensor &mha_out);

void softmax_op(Tensor &input);

void swiglu_op(Tensor &input1, Tensor &input2, Tensor &output);

void embedding_op(const Tensor &weight, Tensor &input, Tensor &output);
}  // namespace CPU_OP