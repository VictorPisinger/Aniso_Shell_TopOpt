#pragma once

#include "eigen3/Eigen/Dense"

class ElementIntegrator {

public:
  ElementIntegrator(const double E = 1.0, const double nu = 0.3) : youngs_module(E), poissons_ratio(nu){};

  Eigen::Matrix<double, 12, 12> compute_element_stiffness(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                          const std::array<bool, 3> flipRotations,
                                                          const double thickness) const;

  std::pair<std::array<double, 6>, std::array<double, 6>>
  compute_element_stress(const std::array<Eigen::Vector3d, 3> globalCoords, const std::array<bool, 3> flipRotations,
                         const double thickness, Eigen::Vector<double, 12> u) const;

  std::array<double, 2> compute_bending_and_membrane_compliance(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                                const std::array<bool, 3> flipRotations,
                                                                const double thickness,
                                                                Eigen::Vector<double, 12> u) const;

  double compute_vm_stress(const std::array<Eigen::Vector3d, 3> globalCoords, const std::array<bool, 3> flipRotations,
                           const double thickness, Eigen::Vector<double, 12> u) const;

  Eigen::Vector3d compute_pressure_force(const std::array<Eigen::Vector3d, 3> globalCoords,
                                         const double pressure) const;

  double compute_thickness_gradient(const std::array<Eigen::Vector3d, 3> globalCoords,
                                    const std::array<bool, 3> flipRotations, const double thickness,
                                    Eigen::Vector<double, 12> u) const;

  double compute_thickness_gradient_auto(const std::array<Eigen::Vector3d, 3> globalCoords,
                                         const std::array<bool, 3> flipRotations, const double thickness,
                                         Eigen::Vector<double, 12> u) const;

  std::array<double, 9> compute_shape_gradients(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                const std::array<bool, 3> flipRotations, const double thickness,
                                                Eigen::Vector<double, 12> u) const;

  std::array<double, 9> compute_pressure_gradients(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                   const double pressure, Eigen::Vector<double, 12> u) const;

private:
  // material parameters
  const double youngs_module;
  const double poissons_ratio;
};
