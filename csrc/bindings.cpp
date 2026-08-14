#include <torch/extension.h>

#include <map>
#include <string>
#include <vector>

void smoke_fill(at::Tensor tensor, double value);
std::map<std::string, std::string> smoke_build_info();

std::vector<at::Tensor> rope_forward(at::Tensor q, at::Tensor k, at::Tensor positions,
                                     at::Tensor cos, at::Tensor sin);
void kv_append(at::Tensor k_rot, at::Tensor v, at::Tensor positions,
               at::Tensor request_indices, at::Tensor k_cache, at::Tensor v_cache);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("smoke_fill", &smoke_fill, py::arg("tensor"), py::arg("value"),
        "Fill a contiguous CUDA tensor in place; exercises launch + dtype dispatch.");
  m.def("build_info", &smoke_build_info,
        "Compile-time and runtime facts about this binary.");

  m.def("rope_forward", &rope_forward, py::arg("q"), py::arg("k"), py::arg("positions"),
        py::arg("cos"), py::arg("sin"),
        "Split-half RoPE over strided Q and K in one launch; returns (q_rot, k_rot).");
  m.def("kv_append", &kv_append, py::arg("k_rot"), py::arg("v"), py::arg("positions"),
        py::arg("request_indices"), py::arg("k_cache"), py::arg("v_cache"),
        "Scatter rotated K and raw V into a contiguous token-major cache, in place.");
}
