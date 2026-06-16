// SPDX-License-Identifier: MIT
#include "peripheral_factory.hpp"

#include <stdexcept>

namespace bldcsim {

std::unique_ptr<IGateDriver> make_gate_driver(const std::string& name,
                                              const Drv8301Config& cfg) {
  if (name == "drv8301") return std::make_unique<Drv8301>(cfg);
  if (name == "drv8302") {
    // Same family / same board mechanicals (Rds_on, dead-time floor, EN ready,
    // thermal), but hardware-configured: OC threshold/mode and the 8 V UVLO
    // come from the DRV8302 datasheet rather than SPI registers.
    Drv8302Config c2;
    c2.en_gate_ready_time_s = cfg.en_gate_ready_time_s;
    c2.dead_time_floor_s = cfg.dead_time_floor_s;
    c2.noctw_pulse_s = cfg.noctw_pulse_s;
    c2.rds_on_ohm = cfg.rds_on_ohm;
    c2.otw_c = cfg.otw_c;
    c2.otsd_c = cfg.otsd_c;
    return std::make_unique<Drv8302>(c2);
  }
  if (name == "drv8323rs") {
    // External-FET SPI driver, same frame family as the DRV8301. Shared board
    // mechanicals carry over; OCP/CSA/UVLO are DRV8323 datasheet defaults.
    Drv8323Config c2;
    c2.dead_time_floor_s = cfg.dead_time_floor_s;
    c2.noctw_pulse_s = cfg.noctw_pulse_s;
    c2.rds_on_ohm = cfg.rds_on_ohm;  // external power-FET Rds_on (CSD-class)
    c2.otw_c = cfg.otw_c;
    c2.otsd_c = cfg.otsd_c;
    return std::make_unique<Drv8323>(c2);
  }
  if (name == "drv8316r") {
    // Integrated FETs + integrated CSA. Datasheet protection/envelope; the
    // shared dead-time floor carries over. The integrated CSA transfer lives
    // in the FeedbackChain (kIntegratedDriverCsa), not here.
    Drv8316rConfig c2;
    c2.dead_time_floor_s = cfg.dead_time_floor_s;
    c2.noctw_pulse_s = cfg.noctw_pulse_s;
    return std::make_unique<Drv8316r>(c2);
  }
  throw std::runtime_error("unknown gate driver platform: " + name);
}

std::unique_ptr<ICurrentAdc> make_current_adc(
    const std::string& name, const Mcp3208Config& cfg,
    std::function<double(int)> source) {
  if (name == "mcp3208")
    return std::make_unique<Mcp3208>(cfg, std::move(source));
  throw std::runtime_error("unknown current-ADC platform: " + name);
}

std::unique_ptr<IAngleSensor> make_angle_sensor(const std::string& name,
                                                const As5600Config& cfg) {
  if (name == "as5600") return std::make_unique<As5600>(cfg);
  if (name == "as5047p") {
    // SPI 14-bit angle with DAEC. The portable mounting/measurement
    // nonidealities carry over from the shared encoder config; the resolution
    // and DAEC latencies come from the AS5047P datasheet (DS000324).
    As5047pConfig c2;
    c2.mounting_offset_rad = cfg.mounting_offset_rad;
    c2.eccentricity_e1_rad = cfg.eccentricity_e1_rad;
    c2.eccentricity_phi1_rad = cfg.eccentricity_phi1_rad;
    c2.eccentricity_e2_rad = cfg.eccentricity_e2_rad;
    c2.eccentricity_phi2_rad = cfg.eccentricity_phi2_rad;
    c2.angle_noise_lsb = cfg.angle_noise_lsb;
    c2.noise_seed = cfg.noise_seed;
    return std::make_unique<As5047p>(c2);
  }
  throw std::runtime_error("unknown angle-sensor platform: " + name);
}

}  // namespace bldcsim
