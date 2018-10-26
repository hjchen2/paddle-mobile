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

#include "../test_helper.h"
#include "../test_include.h"
#include "operators/conv_op.h"

namespace paddle_mobile {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Itype, typename Otype>
void conv2d(const framework::Tensor *input, const framework::Tensor *filter,
            const framework::AttributeMap &attrs, framework::Tensor *output) {
  framework::AttrReader attr_reader(attrs);
  std::vector<int> paddings = attr_reader.Get<std::vector<int>>("paddings");
  std::vector<int> strides = attr_reader.Get<std::vector<int>>("strides");
  std::vector<int> dilations = attr_reader.Get<std::vector<int>>("dilations");
  int groups = attr_reader.Get<int>("groups");
  int kernel_h = filter->dims()[2];
  int kernel_w = filter->dims()[3];
  int pad_h = paddings[0];
  int pad_w = paddings[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int dilation_h = dilations[0];
  int dilation_w = dilations[1];
  auto in_shape = input->dims();
  auto out_shape = output->dims();

  const bool has_depth = 0;
  int kernel_d, pad_d, stride_d, dilation_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
    dilation_d = dilation_h;
  } else {
    kernel_d = stride_d = dilation_d = 1;
    pad_d = 0;
  }
  // Groups
  int o_g = out_shape[1] / groups;
  int k_g = in_shape[1] / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(4 + has_depth);
  vector<int> in_offset(4 + has_depth);
  vector<int> out_offset(4 + has_depth);
  auto offset = [](const framework::Tensor *input, const vector<int> &indics) {
    framework::DDim shape = input->dims();
    size_t count = 0;
    for (int i = 0; i < indics.size(); ++i) {
      count *= shape[i];
      count += indics[i];
    }
    return count;
  };

  const Itype *in_data = input->data<Itype>();
  const Itype *w_data = filter->data<Itype>();
  Otype *out_data = output->mutable_data<Otype>();
  memset(out_data, 0, output->numel() * sizeof(Otype));
  for (int n = 0; n < out_shape[0]; n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out_shape[2] : 1); z++) {
            for (int y = 0; y < out_shape[2 + has_depth]; y++) {
              for (int x = 0; x < out_shape[3 + has_depth]; x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r * dilation_d;
                      int in_y = y * stride_h - pad_h + p * dilation_h;
                      int in_x = x * stride_w - pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (has_depth ? in_shape[2] : 1) &&
                          in_y >= 0 && in_y < in_shape[2 + has_depth] &&
                          in_x >= 0 && in_x < in_shape[3 + has_depth]) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) {
                          weight_offset[2] = r;
                        }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) {
                          in_offset[2] = in_z;
                        }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) {
                          out_offset[2] = z;
                        }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;

                        out_data[offset(output, out_offset)] +=
                            in_data[offset(input, in_offset)] *
                            w_data[offset(filter, weight_offset)];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename Itype, typename Otype, int Kernel, int Pad, int Stride>
int TestConvOp() {
  int kernel_h = Kernel;
  int kernel_w = Kernel;
  int pad_h = Pad;
  int pad_w = Pad;
  int stride_h = Stride;
  int stride_w = Stride;
  int dilation_h = 1;
  int dilation_w = 1;

  int batch_size = 1;
  int input_c = 3;
  int input_h = 100;
  int input_w = 100;
  int output_c = 10;
  framework::DDim input_shape =
      framework::make_ddim({batch_size, input_c, input_h, input_w});
  framework::DDim filter_shape =
      framework::make_ddim({output_c, input_c, kernel_h, kernel_w});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["Input"] = std::vector<std::string>({"input"});
  inputs["Filter"] = std::vector<std::string>({"filter"});
  outputs["Output"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(input, input_shape, -20, 20);

  auto filter_var = scope.get()->Var("filter");
  auto filter = filter_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(filter, filter_shape, -20, 20);

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["strides"].Set<vector<int>>(std::vector<int>({stride_h, stride_w}));
  attrs["paddings"].Set<vector<int>>(std::vector<int>({pad_h, pad_w}));
  attrs["dilations"].Set<vector<int>>(
      std::vector<int>({dilation_h, dilation_w}));
  attrs["groups"].Set<int>(1);

  auto *op = new operators::ConvOp<CPU, float>("conv2d", inputs, outputs, attrs,
                                               scope);
  //  struct timespec ts_begin, ts_end;
  op->InferShape();
  // warmup
  //  op->Run();
  //  clock_gettime(CLOCK_MONOTONIC, &ts_begin);
  //  for (int i = 0; i < 10; ++i) {
  op->Run();
  //  }
  //  clock_gettime(CLOCK_MONOTONIC, &ts_end);
  //  uint64_t elapsed = (ts_end.tv_sec - ts_begin.tv_sec) * 1e3 +
  //                     (ts_end.tv_nsec - ts_begin.tv_nsec) / 1e6;
  //  LOG(kLOG_INFO) << "elapsed: " << elapsed / 10.0 << " ms";

  int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
  int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
  int output_h = (input_h + 2 * pad_h - kernel_extent_h) / stride_h + 1;
  int output_w = (input_w + 2 * pad_w - kernel_extent_w) / stride_w + 1;
  auto output_shape = framework::make_ddim(
      std::vector<int>({batch_size, output_c, output_h, output_w}));
  framework::Tensor output_cmp;
  output_cmp.mutable_data<Otype>(output_shape);
  conv2d<Itype, Otype>(input, filter, attrs, &output_cmp);

  // compare results
  auto output = output_var->template Get<framework::LoDTensor>();
  const Otype *output_data = output->data<Otype>();
  Otype *output_cmp_data = output_cmp.data<Otype>();
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
                          "output[%d] = %d, output_cmp[%d] = %d", i,
                          output_data[i], i, output_cmp_data[i]);
  }
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main() {
  // kernel = 7, pad = 0, stride = 2
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=0, stride=2";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 0, 2>();

  // kernel = 7, pad = 1, stride = 2
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=1, stride=2";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 1, 2>();

  // kernel = 7, pad = 3, stride = 2
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=3, stride=2";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 3, 2>();

  // kernel = 7, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=0, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 0, 1>();

  // kernel = 7, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=1, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 1, 1>();

  // kernel = 7, pad = 3, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=3, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 3, 1>();

  // kernel = 7, pad = 5, stride = 3
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=5, stride=3";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 5, 3>();

  // kernel = 7, pad = 3, stride = 4
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=7, pad=3, stride=4";
  paddle_mobile::TestConvOp<int8_t, int32_t, 7, 3, 4>();
  LOG(paddle_mobile::kLOG_INFO) << "\n";

  // kernel = 3, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=3, pad=0, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 0, 1>();
  // kernel = 3, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "float, kernel=3, pad=0, stride=1";
  paddle_mobile::TestConvOp<float, float, 3, 0, 1>();
  LOG(paddle_mobile::kLOG_INFO) << "\n";

  // kernel = 3, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=3, pad=1, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 3, 1, 1>();
  // kernel = 3, pad = 1, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "float, kernel=3, pad=1, stride=1";
  paddle_mobile::TestConvOp<float, float, 3, 1, 1>();
  LOG(paddle_mobile::kLOG_INFO) << "\n";

  // kernel = 5, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=5, pad=0, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 5, 0, 1>();
  // kernel = 5, pad = 0, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "float, kernel=5, pad=0, stride=1";
  paddle_mobile::TestConvOp<float, float, 5, 0, 1>();
  LOG(paddle_mobile::kLOG_INFO) << "\n";

  // kernel = 5, pad = 2, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "int8, kernel=5, pad=2, stride=1";
  paddle_mobile::TestConvOp<int8_t, int32_t, 5, 2, 1>();
  // kernel = 5, pad = 2, stride = 1
  LOG(paddle_mobile::kLOG_INFO) << "float, kernel=5, pad=2, stride=1";
  paddle_mobile::TestConvOp<float, float, 5, 2, 1>();
}
