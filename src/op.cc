#include "op.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "tensor.h"

namespace CPU_OP {
void rmsnorm_op(const Tensor &weight, const Tensor &input, Tensor &output) {
  const int32_t len = input.size();

  const float *x_ptr = input.ptr<float>();
  const float *w_ptr = weight.ptr<float>();
  float *o_ptr = output.ptr<float>();

  arma::fvec x(const_cast<float *>(x_ptr), len, false, true);
  arma::fvec w(const_cast<float *>(w_ptr), len, false, true);
  arma::fvec o(o_ptr, len, false, true);

  const float eps = 1e-6f;  // TODO 这个超参数来源

  // mean:1/N * (平方和)
  // as_scalar 获取单个元素矩阵的标量值 [1.2] ==> 1.2
  // float rms_x = std::sqrt(arma::as_scalar(arma::mean(arma::pow(x, 2))));
  float rms_x = arma::as_scalar(arma::mean(arma::pow(x, 2))) + eps;
  // %逐元素相乘
  float rsqrt = 1.0f / (std::sqrt(rms_x));
  o = w % (rsqrt * x);
}

void matmul_op(const Tensor &weight, const Tensor &input, Tensor &output, float scale) {
  int32_t ne00 = 1, ne01 = 1;
  int32_t ne10 = 1, ne11 = 1;
  ne00 = input.shape().at(0);
  if (input.shape().size() == 2) {
    ne01 = input.shape().at(1);
  }
  if (weight.shape().size() != 2) {
    fprintf(stderr, "weight shape not 2\n");
    exit(-1);
  }
  ne10 = weight.shape().at(0);
  ne11 = weight.shape().at(1);

  if (ne00 != ne11) {
    fprintf(stderr, "mat shape can't mul\n");
    exit(-1);
  }
  // (row,col)
  // (dim,dim) * (dim,1) ==> (dim,1)
  if (output.size() != ne10 * ne01) {
    fprintf(stderr, "output shape is err,size:(%ld)--(%d,%d)\n", output.size(), ne10, ne01);
    exit(-1);
  }
  // weight是const，只能用const承接
  const float *w_ptr = weight.ptr<float>();
  const float *x_ptr = input.ptr<float>();
  float *o_ptr = output.ptr<float>();

  // o = W*x
  arma::fmat x(const_cast<float *>(x_ptr), ne01, ne00, false, true);
  arma::fmat w(const_cast<float *>(w_ptr), ne11, ne10, false, true);
  arma::fmat o(o_ptr, ne01, ne10, false, true);

  o = x * w;  // 矩阵乘法不具有交换律
  if (std::fabs(scale - 1.0f) > 1e-5f) o *= scale;
}
void matadd_op(const Tensor &input1, const Tensor &input2, Tensor &output) {
  int32_t len = input1.size();
  const float *x_ptr = input1.ptr<float>();
  const float *y_ptr = input2.ptr<float>();
  float *o_ptr = output.ptr<float>();
  arma::fvec x(const_cast<float *>(x_ptr), len, false, true);
  arma::fvec y(const_cast<float *>(y_ptr), len, false, true);
  arma::fvec o(o_ptr, len, false, true);
  o = x + y;
}

void rope_op(Tensor &query, Tensor &key, const Tensor &t_pos, const Tensor &fsin, const Tensor &fcos) {
  /*
  MHA:
      |head_size|head_size|head_size|head_size|
      head_size = 64
      fsin：{32768, 64/2}
  */
  int32_t freq_cache_size = fsin.shape()[1];
  int32_t head_size = freq_cache_size * 2;
  int32_t pos = *t_pos.ptr<int32_t>();
  int32_t dim = query.size();
  int32_t offset;
  float fs, fc;
  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t group_idx = i % head_size; group_idx < head_size / 2; group_idx += 1) {
      offset = pos * freq_cache_size + group_idx;
      fs = *fsin.ptr<float>(offset);
      fc = *fcos.ptr<float>(offset);

      for (int32_t j = 0; j < 2; j++) {
        float *vec = const_cast<float *>(j == 0 ? query.ptr<float>() : key.ptr<float>());
        float v0 = vec[i + group_idx];
        float v1 = vec[i + group_idx + head_size / 2];
        vec[i + group_idx] = fc * v0 - fs * v1;
        vec[i + group_idx + head_size / 2] = fs * v0 + fc * v1;
      }
    }
  }
}

/*
    通过输入的Q,与历史和当前的K1,K2,K3...相乘等到score
    score与历史和当前的V1,V2,V3...相乘得到注意力 QK1*V1 + QK1*V2 + ...(V1,V2维度维度是head_size)
*/
void mha_op(int32_t layer, int32_t pos, int32_t mem_num, int32_t head_num, int32_t head_size, Tensor &query,
            Tensor &k_cache, Tensor &v_cache, Tensor &score, Tensor &mha_out) {
  int32_t ctx_len = score.shape()[1];
  int32_t kv_dim = k_cache.shape()[2];
  int32_t offset = layer * ctx_len * kv_dim;
  float scale = 1.0f / std::sqrt(head_size);
  for (int h = 0; h < head_num; h++) {
    float *q_ptr = query.ptr<float>(h * head_size);
    float *score_ptr = score.ptr<float>(h * ctx_len);
    Tensor q_mat(DataType::kDataTypeFp32, {head_size}, nullptr, q_ptr);
    // 计算 Q*(K1,K2...)
    // config.h中关于kv_dim的描述
    for (int t = 0; t <= pos; t++) {
      float *k_ptr = k_cache.ptr<float>(offset + t * kv_dim + (h / mem_num) * head_size);
      Tensor k_mat(DataType::kDataTypeFp32, {1, head_size}, nullptr, k_ptr);

      Tensor score_mat(DataType::kDataTypeFp32, {1}, nullptr, score_ptr + t);
      CPU_OP::matmul_op(k_mat, q_mat, score_mat, scale);  // k_mat为权重
    }

    // softmax Q*(K1,K2...)
    Tensor score_mat(DataType::kDataTypeFp32, {pos + 1}, nullptr, score_ptr);
    CPU_OP::softmax_op(score_mat);

    // 接下来需要计算 QK1 * V1 + QK1 * V2 + ...(pos+1)个
    float *mha_ptr = mha_out.ptr<float>(h * head_size);
    std::memset(mha_ptr, 0, sizeof(float) * head_size);
    int32_t v_offset = offset + (h / mem_num) * head_size;
    float *v_ptr = v_cache.ptr<float>(v_offset);
    arma::fvec scale_vec(score_mat.ptr<float>(), score.size(), false, true);
    arma::fvec out_vec(mha_ptr, head_size, false, true);
    for (int i = 0; i <= pos; i++) {
      arma::fvec v_vec(v_ptr + i * kv_dim, head_size, false, true);
      out_vec += scale_vec[i] * v_vec;
    }
  }
}

void softmax_op(Tensor &input) {
  size_t size = input.size();
  float max_val = *std::max_element(input.ptr<float>(), input.ptr<float>(size));
  arma::fvec i_mat(input.ptr<float>(), size, false, true);
  i_mat = arma::exp(i_mat - max_val);
  float sum = arma::sum(i_mat);
  i_mat /= sum;
}

void swiglu_op(Tensor &input1, Tensor &input2, Tensor &output) {
  arma::fvec i1_vec(input1.ptr<float>(), input1.size(), false, true);
  arma::fvec i2_vec(input2.ptr<float>(), input2.size(), false, true);
  arma::fvec o_vec(output.ptr<float>(), output.size(), false, true);

  // x = x*sigmoid(x);
  i1_vec %= (1.0f / (1 + arma::exp(-i1_vec)));
  // 逐元素乘
  o_vec = i1_vec % i2_vec;
}

void embedding_op(const Tensor &weight, Tensor &input, Tensor &output) {
  int32_t token_nums = input.size();
  int32_t vocab_size = weight.shape()[0];
  int32_t dim = weight.shape()[1];

  for (int32_t i = 0; i < token_nums; i++) {
    int32_t token = *input.ptr<int32_t>(i);
    if (token >= vocab_size) {
      fprintf(stderr, "token >= vocab_size\n");
      exit(-1);
    }
    float *dest = output.ptr<float>(i * dim);
    const float *src = weight.ptr<float>(token * dim);
    memcpy(dest, src, dim * sizeof(float));
  }
}
}  // namespace CPU_OP

// TODO:算子写测试
// 算子声明的参数