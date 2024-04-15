#include <benchmark/benchmark.h>

#include "elementIntegrator.h"

static void BM_Analytic_dthk(benchmark::State& state) {

  ElementIntegrator ei;

  Eigen::Vector3d p1                                = {0, 0, 0};
  Eigen::Vector3d p2                                = {1, 0, 0};
  Eigen::Vector3d p3                                = {0, 0, 1};
  const std::array<Eigen::Vector3d, 3> globalCoords = {p1, p2, p3};
  const std::array<bool, 3> flipRotations           = {true, false, true};
  const double thickness                            = 0.1;
  const Eigen::Vector<double, 12> u                 = Eigen::Vector<double, 12>::Ones();

  for (auto _ : state)
    benchmark::DoNotOptimize(ei.compute_thickness_gradient(globalCoords, flipRotations, thickness, u));
}
// Register the function as a benchmark
BENCHMARK(BM_Analytic_dthk);

static void BM_Automatic_dthk(benchmark::State& state) {
  ElementIntegrator ei;

  Eigen::Vector3d p1                                = {0, 0, 0};
  Eigen::Vector3d p2                                = {1, 0, 0};
  Eigen::Vector3d p3                                = {0, 0, 1};
  const std::array<Eigen::Vector3d, 3> globalCoords = {p1, p2, p3};
  const std::array<bool, 3> flipRotations           = {true, false, true};
  const double thickness                            = 0.1;
  const Eigen::Vector<double, 12> u                 = Eigen::Vector<double, 12>::Ones();

  for (auto _ : state)
    benchmark::DoNotOptimize(ei.compute_thickness_gradient_auto(globalCoords, flipRotations, thickness, u));
}
BENCHMARK(BM_Automatic_dthk);

static void BM_KE_element(benchmark::State& state) {
  ElementIntegrator ei;

  Eigen::Vector3d p1                                = {0, 0, 0};
  Eigen::Vector3d p2                                = {1, 0, 0};
  Eigen::Vector3d p3                                = {0, 0, 1};
  const std::array<Eigen::Vector3d, 3> globalCoords = {p1, p2, p3};
  const std::array<bool, 3> flipRotations           = {true, false, true};
  const double thickness                            = 0.1;

  for (auto _ : state)
    benchmark::DoNotOptimize(ei.compute_element_stiffness(globalCoords, flipRotations, thickness));
}
BENCHMARK(BM_KE_element);

static void BM_Stress(benchmark::State& state) {
  ElementIntegrator ei;

  Eigen::Vector3d p1                                = {0, 0, 0};
  Eigen::Vector3d p2                                = {1, 0, 0};
  Eigen::Vector3d p3                                = {0, 0, 1};
  const std::array<Eigen::Vector3d, 3> globalCoords = {p1, p2, p3};
  const std::array<bool, 3> flipRotations           = {true, false, true};
  const double thickness                            = 0.1;
  const Eigen::Vector<double, 12> u                 = Eigen::Vector<double, 12>::Ones();

  for (auto _ : state)
    benchmark::DoNotOptimize(ei.compute_element_stress(globalCoords, flipRotations, thickness, u));
}
BENCHMARK(BM_Stress);

static void BM_position_gradient(benchmark::State& state) {
  ElementIntegrator ei;

  Eigen::Vector3d p1                                = {0, 0, 0};
  Eigen::Vector3d p2                                = {1, 0, 0};
  Eigen::Vector3d p3                                = {0, 0, 1};
  const std::array<Eigen::Vector3d, 3> globalCoords = {p1, p2, p3};
  const std::array<bool, 3> flipRotations           = {true, false, true};
  const double thickness                            = 0.1;
  const Eigen::Vector<double, 12> u                 = Eigen::Vector<double, 12>::Ones();

  for (auto _ : state)
    benchmark::DoNotOptimize(ei.compute_shape_gradients(globalCoords, flipRotations, thickness, u));
}
BENCHMARK(BM_position_gradient);

BENCHMARK_MAIN();
