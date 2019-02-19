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

#pragma once

#include "operators/math/gemm/gemm_kernel.h"

namespace paddle_mobile {
namespace operators {
namespace math {

struct SgemmStrategy {
  typedef float Itype;
  typedef float Otype;

  typedef void (*kern_type)(const Itype *, const Itype *, const int, Otype *);
  kern_type kernel;

  static int out_width() { return 8; }

  static int out_height() {
#ifdef __aarch64__
    return 12;
#else
    return 6;
#endif
  }

  SgemmStrategy() {
#ifdef __aarch64__
    kernel = sgemm_12x8;
#else
    kernel = sgemm_6x8;
#endif
  }
};

struct I8o32gemmStrategy {
  typedef int8_t Itype;
  typedef int32_t Otype;

  typedef void (*kern_type)(const Itype *, const Itype *, const int, Otype *);
  kern_type kernel;

  static int out_width() { return 8; }

  static int out_height() {
#ifdef __aarch64__
    return 12;
#else
    return 6;
#endif
  }

  I8o32gemmStrategy() {}
};

struct SgemvStrategy {
  typedef float Itype;
  typedef float Otype;

  typedef void (*kern_type)(const Itype *, const Itype *, const int, Otype *);
  kern_type kernel;

  static int out_width() { return 1; }

  static int out_height() {
#ifdef __aarch64__
    return 12;
#else
    return 6;
#endif
  }
};

struct I8o32gemvStrategy {
  typedef int8_t Itype;
  typedef int32_t Otype;

  typedef void (*kern_type)(const Itype *, const Itype *, const int, Otype *);
  kern_type kernel;

  static int out_width() { return 1; }

  static int out_height() {
#ifdef __aarch64__
    return 12;
#else
    return 6;
#endif
  }
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
