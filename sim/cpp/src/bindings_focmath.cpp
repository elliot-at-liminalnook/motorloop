// SPDX-License-Identifier: MIT
// pybind module exposing the combinational FOC math harness (rtl/foc_math.v)
// so the RTL primitives can be checked bit-for-bit against
// sim/scripts/foc_reference.py (sim/tests/test_foc_math.py).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <tuple>

#include "Vfoc_math.h"
#include "verilated.h"

namespace py = pybind11;

namespace {

// Sign-extend an 18-bit two's-complement value held in a uint32.
std::int32_t sx18(std::uint32_t v) {
  v &= 0x3FFFF;
  return (v & 0x20000) ? static_cast<std::int32_t>(v | 0xFFFC0000) : v;
}

// Sign-extend a 16-bit value.
std::int32_t sx16(std::uint32_t v) {
  v &= 0xFFFF;
  return (v & 0x8000) ? static_cast<std::int32_t>(v | 0xFFFF0000) : v;
}

std::uint32_t mask18(std::int32_t v) {
  return static_cast<std::uint32_t>(v) & 0x3FFFF;
}

class FocMath {
 public:
  FocMath() : ctx_(std::make_unique<VerilatedContext>()),
              dut_(std::make_unique<Vfoc_math>(ctx_.get())) {}

  // Evaluate the combinational harness for the given inputs and return all
  // outputs as a dict-friendly tuple.
  py::dict eval(int theta, int ia, int ib, int vd, int vq,
                int valpha_in, int vbeta_in) {
    dut_->theta = static_cast<std::uint32_t>(theta) & 0xFFFF;
    dut_->ia = mask18(ia);
    dut_->ib = mask18(ib);
    dut_->vd = mask18(vd);
    dut_->vq = mask18(vq);
    dut_->valpha_in = mask18(valpha_in);
    dut_->vbeta_in = mask18(vbeta_in);
    dut_->eval();

    std::uint64_t d3 = dut_->duty3;
    py::dict out;
    out["sin"] = sx16(dut_->sin_out);
    out["cos"] = sx16(dut_->cos_out);
    out["ialpha"] = sx18(dut_->ialpha);
    out["ibeta"] = sx18(dut_->ibeta);
    out["id"] = sx18(dut_->id);
    out["iq"] = sx18(dut_->iq);
    out["valpha"] = sx18(dut_->valpha_out);
    out["vbeta"] = sx18(dut_->vbeta_out);
    out["duty_a"] = static_cast<int>(d3 & 0xFFFF);
    out["duty_b"] = static_cast<int>((d3 >> 16) & 0xFFFF);
    out["duty_c"] = static_cast<int>((d3 >> 32) & 0xFFFF);
    return out;
  }

 private:
  std::unique_ptr<VerilatedContext> ctx_;
  std::unique_ptr<Vfoc_math> dut_;
};

}  // namespace

PYBIND11_MODULE(focmath, m) {
  m.doc() = "Combinational FOC math harness (RTL parity testing)";
  py::class_<FocMath>(m, "FocMath")
      .def(py::init<>())
      .def("eval", &FocMath::eval, py::arg("theta"), py::arg("ia"),
           py::arg("ib"), py::arg("vd"), py::arg("vq"),
           py::arg("valpha_in"), py::arg("vbeta_in"));
}
