#pragma once

#include <array>
#include <cmath>

namespace bldcsim {

// Lumped thermal model (realism stage 4): per-FET junction RC to ambient,
// a DRV8301 die lump, and a motor-winding lump. Loss inputs come from the
// plant (conduction) and the bench (switching edges); outputs are
// temperatures plus the drift multipliers the plant consumes. OTW/OTSD
// become emergent through the DRV die temperature.
//
// Crude by design: datasheet RthJC + assumed board factors (provenance
// 'assumed' in params.toml). Magnet temperature is approximated by the
// winding lump for the Ke derating.

struct ThermalConfig {
  bool enabled = false;
  double fet_rth_jc_k_w = 0.72;
  double fet_rth_ca_k_w = 20.0;
  double fet_cth_j_k = 0.8;
  double drv_rth_ja_k_w = 30.0;
  double drv_cth_j_k = 0.3;
  double motor_rth_wa_k_w = 8.0;
  double motor_cth_j_k = 15.0;
  double sw_loss_k_j_va = 50e-9;
  double gate_drive_e_per_edge_j = 0.6e-6;
  double drv_quiescent_w = 0.15;
  double alpha_cu_1_k = 3.93e-3;
  double ke_derate_1_k = 1.1e-3;
  double rds_tempco_1_k = 4.0e-3;
  double ambient_c = 25.0;
};

class ThermalModel {
 public:
  explicit ThermalModel(const ThermalConfig& config) : config_(config) {
    for (auto& t : fet_tj_c_) t = config_.ambient_c;
    drv_t_c_ = config_.ambient_c;
    motor_t_c_ = config_.ambient_c;
  }

  // Called by the bench on every gate edge of leg k with the instantaneous
  // bus voltage and phase current: accumulates switching energy.
  void add_switch_edge(int leg, double vbus_v, double i_a) {
    if (!config_.enabled) return;
    pending_sw_j_[leg] += config_.sw_loss_k_j_va * vbus_v * std::abs(i_a);
    pending_drv_j_ += config_.gate_drive_e_per_edge_j;
  }

  // Advance by dt with the plant's per-leg conduction losses and the total
  // winding copper loss.
  void update(double dt_s, const std::array<double, 3>& leg_conduction_w,
              double winding_w) {
    if (!config_.enabled || dt_s <= 0.0) return;
    const double rth_fet = config_.fet_rth_jc_k_w + config_.fet_rth_ca_k_w;
    for (int k = 0; k < 3; ++k) {
      const double p = leg_conduction_w[k] + pending_sw_j_[k] / dt_s;
      pending_sw_j_[k] = 0.0;
      const double dT = (p - (fet_tj_c_[k] - config_.ambient_c) / rth_fet) /
                        config_.fet_cth_j_k;
      fet_tj_c_[k] += dT * dt_s;
    }
    const double p_drv = config_.drv_quiescent_w + pending_drv_j_ / dt_s;
    pending_drv_j_ = 0.0;
    drv_t_c_ += (p_drv - (drv_t_c_ - config_.ambient_c) /
                             config_.drv_rth_ja_k_w) /
                config_.drv_cth_j_k * dt_s;
    motor_t_c_ += (winding_w - (motor_t_c_ - config_.ambient_c) /
                                   config_.motor_rth_wa_k_w) /
                  config_.motor_cth_j_k * dt_s;
  }

  double fet_tj_max_c() const {
    return std::max(fet_tj_c_[0], std::max(fet_tj_c_[1], fet_tj_c_[2]));
  }
  double drv_t_c() const { return drv_t_c_; }
  double motor_t_c() const { return motor_t_c_; }

  // Drift multipliers for the plant.
  double r_scale() const {
    return 1.0 + config_.alpha_cu_1_k * (motor_t_c_ - config_.ambient_c);
  }
  double ke_scale() const {
    return 1.0 - config_.ke_derate_1_k * (motor_t_c_ - config_.ambient_c);
  }
  double rds_scale() const {
    return 1.0 + config_.rds_tempco_1_k * (fet_tj_max_c() - config_.ambient_c);
  }

  bool enabled() const { return config_.enabled; }

 private:
  ThermalConfig config_;
  std::array<double, 3> fet_tj_c_{};
  std::array<double, 3> pending_sw_j_{};
  double pending_drv_j_ = 0.0;
  double drv_t_c_ = 25.0;
  double motor_t_c_ = 25.0;
};

}  // namespace bldcsim
