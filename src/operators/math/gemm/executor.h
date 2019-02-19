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

#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common/log.h"
#include "operators/math/gemm/cpu_info.h"
#include "operators/math/gemm/gemm_kernel.h"
// #include "operators/math/gemm/package.h"

#define ALLOC_ROUND 64

namespace paddle_mobile {
namespace operators {
namespace math {

inline int RoundUp(const int &x) {
  return ((x + ALLOC_ROUND - 1) / ALLOC_ROUND) * ALLOC_ROUND;
}

inline int CeilDiv(const int &x, const int &y) { return (x + y - 1) / y; }

class Executor {
 public:
  Executor() : num_threads_(1) {
#ifdef _OPENMP
    num_threads_ = omp_get_num_threads();
#endif
  }
  virtual ~Executor() {}

 protected:
  int num_threads_;
};

template <typename Strategy>
class GemmExecutor : public Executor {
  typedef typename Strategy::Itype Itype;
  typedef typename Strategy::Otype Otype;

 public:
  GemmExecutor(const CPUInfo *info, const bool transA, const bool transB,
               const int M, const int N, const int K)
      : Executor(),
        info_(info),
        transA_(transA),
        transB_(transB),
        M_(M),
        N_(N),
        K_(K) {
    const unsigned int L1_size = info->L1_cache * 9 / 10;  // use 9/10 L1 cache
    const unsigned int L2_size = info->L2_cache * 9 / 10;  // use 9/10 L2 cache

    const int lhs_tile_size = Strategy::out_height() * K * sizeof(Itype);
    const int rhs_tile_size = Strategy::out_width() * K * sizeof(Itype);
    rhs_tile_num_ = CeilDiv(L1_size, rhs_tile_size);
    rhs_tile_num_ = (rhs_tile_num_ > 0) ? rhs_tile_num_ : 1;
    rhs_tile_num_ = rhs_tile_num_ * Strategy::out_width();
    lhs_tile_num_ = num_threads_ * Strategy::out_height();

    lhs_worksize_ = RoundUp(lhs_tile_num_ * K * sizeof(Itype));
    rhs_worksize_ = RoundUp(rhs_tile_num_ * K * sizeof(Itype));
    out_worksize_ =
        RoundUp(Strategy::out_height() * Strategy::out_width() * num_threads_);
    set_working_space();
    DLOG << "rhs_tile_num: " << rhs_tile_num_
         << ", lhs_tile_num: " << lhs_tile_num_;
  }

  void operator()(const float alpha, const Itype *A, const int lda,
                  const Itype *B, const int ldb, const float beta, Otype *C,
                  const int ldc) {
    for (int rhs_start = 0; rhs_start < N_; rhs_start += rhs_tile_num_) {
      int rhs_end = rhs_start + rhs_tile_num_;
      if (rhs_end > N_) {
        rhs_end = N_;
      }
      // load rhs into rhs_workspace
      #pragma omp parallel for
      for (int lhs_tile = 0; lhs_tile < M_;
           lhs_tile += Strategy::out_height()) {
#ifdef _OPENMP
        int thread_id = omp_get_thread_num();
#else
        int thread_id = 0;
#endif
        const Itype *lhs =
            lhs_workspace_ + thread_id * Strategy::out_height() * K_;
        // load lhs into lhs_workspace
        for (int start = 0; start < rhs_end - rhs_start;
             start += Strategy::out_width()) {
          const Itype *rhs = rhs_workspace_ + start * K_;
          Otype *output = out_workspace_ + thread_id * Strategy::out_height() *
                                               Strategy::out_width();
          strategy_.kernel(lhs, rhs, K_, output);
          // write back real output
        }
      }
    }
  }

  void set_working_space() {
    size_t worksize =
        2 * ALLOC_ROUND + lhs_worksize_ + rhs_worksize_ + out_worksize_;

    // TODO(hjchen2): use memory pool
    working_space_ = new int8_t[worksize];
    // we have malloced extra 128 bytes memory
    size_t diff = 0x40 - reinterpret_cast<intptr_t>(working_space_) & 0x3F;

    lhs_workspace_ = reinterpret_cast<Itype *>(working_space_ + diff);
    rhs_workspace_ =
        reinterpret_cast<Itype *>(working_space_ + diff + lhs_worksize_);
    out_workspace_ = reinterpret_cast<Otype *>(working_space_ + diff +
                                               lhs_worksize_ + rhs_worksize_);
  }

  virtual ~GemmExecutor() {
    if (working_space_) delete[] working_space_;
  }

 private:
  const CPUInfo *info_;

  const unsigned int M_;
  const unsigned int N_;
  const unsigned int K_;
  const bool transA_;
  const bool transB_;

  unsigned int lhs_tile_num_ = 0;
  unsigned int rhs_tile_num_ = 0;

  unsigned int lhs_worksize_ = 0;
  unsigned int rhs_worksize_ = 0;
  unsigned int out_worksize_ = 0;

  Itype *lhs_workspace_ = nullptr;
  Itype *rhs_workspace_ = nullptr;
  Otype *out_workspace_ = nullptr;
  int8_t *working_space_ = nullptr;

  Strategy strategy_;
};

template <typename Strategy>
class GemvExecutor : public Executor {
  typedef typename Strategy::Itype Itype;
  typedef typename Strategy::Otype Otype;

 public:
  GemvExecutor(const CPUInfo *info, const bool transA, const int M, const int N)
      : Executor(), info_(info), M_(M), N_(N) {}

  void operator()(const float alpha, const Itype *A, const int lda,
                  const Itype *B, const float beta, Otype *C) {
    //    strategy_.kernel();
  }

  virtual ~GemvExecutor() {}

 private:
  const CPUInfo *const info_;

  const unsigned int M_;
  const unsigned int N_;

  Strategy strategy_;
};

}  // namespace math
}  // namespace operators
}  // namespace paddle_mobile
