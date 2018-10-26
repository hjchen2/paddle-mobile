/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef DEQUANT_OP

#include "operators/kernel/dequantize_kernel.h"

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <>
bool DequantizeKernel<CPU, float>::Init(DequantizeParam<CPU> *param) {
  return true;
}

template <>
void DequantizeKernel<CPU, float>::Compute(
    const DequantizeParam<CPU> &param) const {
  const Tensor *input = param.input_;
  Tensor *output = param.out_;
  float activation_scale = param.activation_scale_->data<float>()[0];
  float weight_scale = param.weight_scale_;
  const int32_t *x = input->data<const int32_t>();
  float *y = output->mutable_data<float>();
  size_t size = output->numel();
  // float scale = 1.f / (activation_scale * weight_scale);
  float scale = activation_scale / weight_scale;
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
  size_t loop = size >> 4;
  size_t remain = size & 0xF;
  float32x4_t s = vdupq_n_f32(scale);
  for (size_t i = 0; i < loop; ++i) {
    int32x4_t r0 = vld1q_s32(x);
    int32x4_t r1 = vld1q_s32(x + 4);
    int32x4_t r2 = vld1q_s32(x + 8);
    int32x4_t r3 = vld1q_s32(x + 12);
    float32x4_t f0 = vcvtq_f32_s32(r0);
    float32x4_t f1 = vcvtq_f32_s32(r1);
    float32x4_t f2 = vcvtq_f32_s32(r2);
    float32x4_t f3 = vcvtq_f32_s32(r3);
    f0 = vmulq_f32(f0, s);
    f1 = vmulq_f32(f1, s);
    f2 = vmulq_f32(f2, s);
    f3 = vmulq_f32(f3, s);
    vst1q_f32(y, f0);
    vst1q_f32(y + 4, f1);
    vst1q_f32(y + 8, f2);
    vst1q_f32(y + 12, f3);
    x += 16;
    y += 16;
  }
  size = remain;
#endif
  for (size_t i = 0; i < size; ++i) {
    y[i] = x[i] * scale;
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
