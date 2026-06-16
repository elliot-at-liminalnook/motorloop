// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>

namespace bldcsim {

// Pin-level inputs a gate-driver model receives from the RTL (generic across
// parts - the same physical pins regardless of register map). Formerly
// Drv8301Inputs; renamed for the role interface (platform-abstraction stage 1).
struct DriverInputs {
  bool en_gate = false;
  bool dc_cal = false;
  std::array<bool, 3> inh{false, false, false};
  std::array<bool, 3> inl{false, false, false};
  bool nscs = true;
  bool sclk = false;
  bool sdi = false;
};

// Role interface for a three-phase gate driver (DRV8301, DRV8316, …). Captures
// the bench<->model call surface at the pin level; register semantics and fault
// behavior live inside each concrete model. The bench holds an IGateDriver and
// a factory picks the concrete part from config.
class IGateDriver {
 public:
  virtual ~IGateDriver() = default;

  // Advance to absolute time with the given pin inputs and phase currents.
  virtual void update(double t_s, const DriverInputs& in,
                      const std::array<double, 3>& phase_currents_a,
                      double pvdd_v, double die_temp_c) = 0;

  // Pin outputs to the RTL.
  virtual const std::array<bool, 3>& gate_high() const = 0;
  virtual const std::array<bool, 3>& gate_low() const = 0;
  virtual bool nfault() const = 0;
  virtual bool noctw() const = 0;
  virtual bool sdo() const = 0;

  // Probes the bench/scenarios read.
  virtual bool pvdd_uv_active() const = 0;
  virtual bool dc_cal_active(int channel) const = 0;
  virtual std::uint16_t reg(int addr) const = 0;
  virtual bool ready() const = 0;
  virtual long frame_errors() const = 0;

  // Fault-injection hooks (no-op by default so a model lacking a given fault
  // simply ignores the injection - keeps the realism scenarios portable).
  virtual void inject_register_reset() {}
  virtual void inject_otw(bool /*active*/) {}
  virtual void inject_latched_fault() {}
};

}  // namespace bldcsim
