// SPDX-License-Identifier: MIT
#pragma once

namespace bldcsim {

// Role interface for the rotor angle sensor (AS5600, AS5047P, …). Captures the
// bench<->model surface for a PWM-output sensor (the AS5600 path); an SPI
// sensor (AS5047) will add an SPI interface in Phase B (RTL variants), since
// the RTL-facing connection differs. The bench holds an IAngleSensor; a factory
// picks the part from config.
class IAngleSensor {
 public:
  virtual ~IAngleSensor() = default;

  // Advance to absolute time with the true mechanical rotor angle.
  virtual void update(double t_s, double theta_mech_rad) = 0;

  // The RTL-facing output pin (AS5600 PWM-encoded angle).
  virtual bool out() const = 0;

  // The sensor's filtered angle estimate (diagnostic, radians).
  virtual double filtered_angle_rad() const = 0;

  // SPI-slave bus, for SPI angle sensors (AS5047P). The bench drives the RTL
  // angle master's pins in; the sensor presents MISO. No-op for PWM parts.
  virtual void spi_io(bool /*cs_n*/, bool /*sclk*/, bool /*mosi*/) {}
  virtual bool miso() const { return false; }

  // Fault-injection (no-op by default).
  virtual void inject_magnet_loss(bool /*lost*/) {}
};

}  // namespace bldcsim
