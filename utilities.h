#include <array>
#include <cmath>

double distance(const std::array<double, 3> &a,
                const std::array<double, 3> &b) {

  const double v0 = a[0] - b[0];
  const double v1 = a[1] - b[1];
  const double v2 = a[2] - b[2];
  return std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
}
