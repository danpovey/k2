/**
 * @brief python wrappers for Ragged<T>.
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <vector>

#include "k2/csrc/ragged.h"
#include "k2/python/csrc/torch/ragged.h"
#include "k2/python/csrc/torch/torch_util.h"
#include "torch/extension.h"

namespace k2 {

template <typename T>
static void PybindRaggedTpl(py::module &m, const char *name) {
  using PyClass = Ragged<T>;
  py::class_<PyClass> pyclass(m, name);

  pyclass.def("values", [](PyClass &self) -> torch::Tensor {
    Array1<T> &values = self.values;
    return ToTensor(values);
  });

  pyclass.def(
      "row_splits",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_splits = self.RowSplits(axis);
        return ToTensor(row_splits);
      },
      py::arg("axis"));

  pyclass.def(
      "row_ids",
      [](PyClass &self, int32_t axis) -> torch::Tensor {
        Array1<int32_t> &row_ids = self.RowIds(axis);
        return ToTensor(row_ids);
      },
      py::arg("axis"));

  pyclass.def("tot_size", &PyClass::TotSize, py::arg("axis"));
  pyclass.def("dim0", &PyClass::Dim0);
  pyclass.def("num_axes", &PyClass::NumAxes);
  pyclass.def("index", &PyClass::Index, py::arg("axis"), py::arg("i"));
}

static void PybindRaggedImpl(py::module &m) {
  PybindRaggedTpl<Arc>(m, "RaggedArc");
}

}  // namespace k2

void PybindRagged(py::module &m) { k2::PybindRaggedImpl(m); }
