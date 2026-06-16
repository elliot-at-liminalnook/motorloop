// SPDX-License-Identifier: MIT
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "as5047p.hpp"
#include "as5600.hpp"
#include "drv8301.hpp"
#include "drv8302.hpp"
#include "drv8316r.hpp"
#include "drv8323.hpp"
#include "i_angle_sensor.hpp"
#include "i_current_adc.hpp"
#include "i_gate_driver.hpp"
#include "mcp3208.hpp"

namespace bldcsim {

// Peripheral factories (platform-abstraction stage 2): construct the concrete
// model selected by name. The current parts (drv8301 / mcp3208 / as5600) are
// the defaults; new parts register here as they are added (Phase B). An unknown
// name throws.
std::unique_ptr<IGateDriver> make_gate_driver(const std::string& name,
                                              const Drv8301Config& cfg);

std::unique_ptr<ICurrentAdc> make_current_adc(
    const std::string& name, const Mcp3208Config& cfg,
    std::function<double(int)> source);

std::unique_ptr<IAngleSensor> make_angle_sensor(const std::string& name,
                                                const As5600Config& cfg);

}  // namespace bldcsim
