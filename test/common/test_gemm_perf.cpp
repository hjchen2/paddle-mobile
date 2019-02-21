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

#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/math/gemm.h"
#include "operators/math/gemm/cblas.h"
#include "operators/math/math_function.h"

template <typename Itype, typename Otype>
void GemmPerf(int M, int N, int K, bool transA, bool transB) {
  Tensor aa, bb, cc;
  auto *aaptr = aa.mutable_data<Itype>({M, K});
  auto *bbptr = bb.mutable_data<Itype>({K, N});
  auto *ccptr = cc.mutable_data<Otype>({M, N});

  // warm-up 10 times
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::MatMul<Itype, Otype>(
        aa, false, bb, false, static_cast<float>(1), &cc, static_cast<float>(0),
        false, nullptr);
  }

  auto time_start0 = time();
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::MatMul<Itype, Otype>(
        aa, false, bb, false, static_cast<float>(1), &cc, static_cast<float>(0),
        false, nullptr);
  }
  auto time_end0 = time();
  std::cout << "gemm cost: " << time_diff(time_start0, time_end0) / 10
            << "ms\n";

  time_start0 = time();
  for (int j = 0; j < 10; ++j) {
    paddle_mobile::operators::math::cblas_sgemm(
        false, false, M, N, K, 1.f, aaptr, K, bbptr, N, 0.f, ccptr, K);
  }
  time_end0 = time();
  std::cout << "gemm cost: " << time_diff(time_start0, time_end0) / 10
            << "ms\n";
}

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(1);

  // float
  std::cout << "float gemm: 128 - 128 - 128 - false - false " << std::endl;
  GemmPerf<float, float>(128, 128, 128, false, false);

  std::cout << "float gemm: 256 - 256 - 256 - false - false " << std::endl;
  GemmPerf<float, float>(256, 256, 256, false, false);

  std::cout << "float gemm: 512 - 512 - 512 - false - false " << std::endl;
  GemmPerf<float, float>(512, 512, 512, false, false);

  std::cout << "float gemm: 1024 - 1024 - 1024 - false - false " << std::endl;
  GemmPerf<float, float>(1024, 1024, 1024, false, false);

  return 0;
}
