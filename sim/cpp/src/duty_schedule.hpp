#pragma once

#include <utility>
#include <vector>

namespace bldcsim {

// Piecewise-constant schedule: value of the last segment whose start time is
// <= t. Mirrors the Python reference runner so trajectories match bit-for-bit
// up to floating-point associativity.
class DutySchedule {
 public:
  explicit DutySchedule(std::vector<std::pair<double, double>> segments)
      : segments_(std::move(segments)) {}

  double at(double t) const {
    double duty = 0.0;
    for (const auto& [t_start, value] : segments_) {
      if (t < t_start) {
        break;
      }
      duty = value;
    }
    return duty;
  }

 private:
  std::vector<std::pair<double, double>> segments_;
};

}  // namespace bldcsim
