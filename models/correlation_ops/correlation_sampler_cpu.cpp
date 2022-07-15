#include <torch/extension.h>
#include <vector>
#include <iostream>

// declarations

torch::Tensor correlation_cpp_forward(
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);

std::vector<torch::Tensor> correlation_cpp_backward(
    torch::Tensor grad_output,
    torch::Tensor input1,
    torch::Tensor input2,
    int kH, int kW,
    int patchH, int patchW,
    int padH, int padW,
    int dilationH, int dilationW,
    int dilation_patchH, int dilation_patchW,
    int dH, int dW);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &correlation_cpp_forward, "Spatial Correlation Sampler Forward");
  m.def("backward", &correlation_cpp_backward, "Spatial Correlation Sampler backward");
}
