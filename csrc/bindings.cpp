#include <torch/extension.h>

#include <map>
#include <string>

void smoke_fill(at::Tensor tensor, double value);
std::map<std::string, std::string> smoke_build_info();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("smoke_fill", &smoke_fill, py::arg("tensor"), py::arg("value"),
        "Fill a contiguous CUDA tensor in place; exercises launch + dtype dispatch.");
  m.def("build_info", &smoke_build_info,
        "Compile-time and runtime facts about this binary.");
}
