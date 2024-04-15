#include "elementIntegrator.h"

#include "shellElement.h"

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/eigen.hpp>

Eigen::Matrix<double, 12, 12>
ElementIntegrator::compute_element_stiffness(const std::array<Eigen::Vector3d, 3> globalCoords,
                                             const std::array<bool, 3> flipRotations, const double thickness) const {

  return Shell::compute_element_stiffness<double>(globalCoords, flipRotations, thickness, this->youngs_module,
                                                  this->poissons_ratio);
}

std::pair<std::array<double, 6>, std::array<double, 6>>
ElementIntegrator::compute_element_stress(const std::array<Eigen::Vector3d, 3> globalCoords,
                                          const std::array<bool, 3> flipRotations, const double thickness,
                                          const Eigen::Vector<double, 12> u) const {

  return Shell::compute_element_stress<double>(globalCoords, flipRotations, thickness, u, this->youngs_module,
                                               this->poissons_ratio);
}

std::array<double, 2>
ElementIntegrator::compute_bending_and_membrane_compliance(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                           const std::array<bool, 3> flipRotations,
                                                           const double thickness, Eigen::Vector<double, 12> u) const {

  return Shell::compute_bending_and_membrane_compliance(globalCoords, flipRotations, thickness, u, this->youngs_module,
                                                        this->poissons_ratio);
}

double ElementIntegrator::compute_vm_stress(const std::array<Eigen::Vector3d, 3> globalCoords,
                                            const std::array<bool, 3> flipRotations, const double thickness,
                                            Eigen::Vector<double, 12> u) const {
  return Shell::compute_vm_stress(globalCoords, flipRotations, thickness, u, this->youngs_module, this->poissons_ratio);
}

Eigen::Vector3d ElementIntegrator::compute_pressure_force(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                          const double pressure) const {
  return Shell::compute_element_pressure<double>(globalCoords, pressure);
}

double ElementIntegrator::compute_thickness_gradient(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                     const std::array<bool, 3> flipRotations, const double thickness,
                                                     Eigen::Vector<double, 12> u) const {

  const auto basis = Shell::compute_local_basis(globalCoords);
  const auto T     = Shell::compute_rotation_matrix(basis);
  const auto area  = Shell::compute_area(globalCoords);

  std::array<Eigen::Vector3d, 3> localCoords;
  localCoords[0] = T * globalCoords[0];
  localCoords[1] = T * globalCoords[1];
  localCoords[2] = T * globalCoords[2];

  Eigen::Matrix<double, 12, 12> Te = Eigen::Matrix<double, 12, 12>::Identity();
  for (int t = 0; t < 3; t++) {
    Te(Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>), Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>)) = T;
  }

  const auto bm = Shell::compute_membrane_strain_interpolation(globalCoords, basis, area);
  const auto bb = Shell::compute_bending_strain_interpolation(localCoords, flipRotations);

  Eigen::Vector<double, 12> u_local = Te * u;
  const auto membrane_strain        = bm * u_local.block<6, 1>(0, 0);
  const auto bending_strain         = bb * u_local.block<6, 1>(6, 0);

  Eigen::Matrix<double, 3, 3> Ebb_dt = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 3> Emm_dt = Eigen::Matrix<double, 3, 3>::Zero();

  Ebb_dt(0, 0) = 1;
  Ebb_dt(1, 1) = 1;
  Ebb_dt(0, 1) = poissons_ratio;
  Ebb_dt(1, 0) = poissons_ratio;
  Ebb_dt(2, 2) = 0.5 - 0.5 * poissons_ratio;
  Ebb_dt *= (3.0 * thickness * thickness * youngs_module) / (12.0 * (1.0 - poissons_ratio * poissons_ratio));

  Emm_dt(0, 0) = 1;
  Emm_dt(1, 1) = 1;
  Emm_dt(0, 1) = poissons_ratio;
  Emm_dt(1, 0) = poissons_ratio;
  Emm_dt(2, 2) = 0.5 - 0.5 * poissons_ratio;
  Emm_dt *= (youngs_module / (1.0 - poissons_ratio * poissons_ratio));

  const double membrane_dt = membrane_strain.transpose() * Emm_dt * membrane_strain;
  const double bending_dt  = bending_strain.transpose() * Ebb_dt * bending_strain;

  return -area * (membrane_dt + bending_dt);
}

double ElementIntegrator::compute_thickness_gradient_auto(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                          const std::array<bool, 3> flipRotations,
                                                          const double thickness, Eigen::Vector<double, 12> u) const {

  autodiff::dual t_ad                    = thickness;
  Eigen::Vector<autodiff::dual, 3> gc_1  = globalCoords[0];
  Eigen::Vector<autodiff::dual, 3> gc_2  = globalCoords[1];
  Eigen::Vector<autodiff::dual, 3> gc_3  = globalCoords[2];
  Eigen::Vector<autodiff::dual, 12> u_ad = u;

  auto f = [&](const Eigen::Vector<autodiff::dual, 3>& c1, const Eigen::Vector<autodiff::dual, 3>& c2,
               const Eigen::Vector<autodiff::dual, 3>& c3, const autodiff::dual& t) {
    const std::array<Eigen::Vector<autodiff::dual, 3>, 3> gc = {c1, c2, c3};

    return Shell::compute_compliance<autodiff::dual>(gc, flipRotations, t, u_ad, this->youngs_module,
                                                     this->poissons_ratio);
  };

  auto dthk = autodiff::derivative(f, autodiff::wrt(t_ad), autodiff::at(gc_1, gc_2, gc_3, t_ad));

  // add negative sign from -u dk u
  return -dthk;
}

std::array<double, 9> ElementIntegrator::compute_shape_gradients(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                                 const std::array<bool, 3> flipRotations,
                                                                 const double thickness,
                                                                 Eigen::Vector<double, 12> u) const {

  // using namespace autodiff;
  autodiff::dual t_ad                    = thickness;
  Eigen::Vector<autodiff::dual, 3> gc_1  = globalCoords[0];
  Eigen::Vector<autodiff::dual, 3> gc_2  = globalCoords[1];
  Eigen::Vector<autodiff::dual, 3> gc_3  = globalCoords[2];
  Eigen::Vector<autodiff::dual, 12> u_ad = u;

  const auto f = [&](const Eigen::Vector<autodiff::dual, 3>& c1, const Eigen::Vector<autodiff::dual, 3>& c2,
                     const Eigen::Vector<autodiff::dual, 3>& c3, const autodiff::dual& t) {
    const std::array<Eigen::Vector<autodiff::dual, 3>, 3> gc = {c1, c2, c3};

    return Shell::compute_compliance<autodiff::dual>(gc, flipRotations, t, u_ad, this->youngs_module,
                                                     this->poissons_ratio);
  };

  const Eigen::VectorXd g =
      -1.0 * autodiff::gradient(f, autodiff::wrt(gc_1, gc_2, gc_3), autodiff::at(gc_1, gc_2, gc_3, t_ad));

  return {g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8)};
}

std::array<double, 9> ElementIntegrator::compute_pressure_gradients(const std::array<Eigen::Vector3d, 3> globalCoords,
                                                                    const double pressure,
                                                                    Eigen::Vector<double, 12> u) const {

  autodiff::dual p_ad                    = pressure;
  Eigen::Vector<autodiff::dual, 3> gc_1  = globalCoords[0];
  Eigen::Vector<autodiff::dual, 3> gc_2  = globalCoords[1];
  Eigen::Vector<autodiff::dual, 3> gc_3  = globalCoords[2];
  Eigen::Vector<autodiff::dual, 12> u_ad = u;

  const auto f = [&](const Eigen::Vector<autodiff::dual, 3>& c1, const Eigen::Vector<autodiff::dual, 3>& c2,
                     const Eigen::Vector<autodiff::dual, 3>& c3) {
    const std::array<Eigen::Vector<autodiff::dual, 3>, 3> gc = {c1, c2, c3};
    const auto fe = Shell::compute_element_pressure<autodiff::dual>(gc, p_ad);

    Eigen::Vector<autodiff::dual, 12> force = {fe(0), fe(0), fe(0), fe(1), fe(1), fe(1),
                                               fe(2), fe(2), fe(2), 0.0,   0.0,   0.0};

    return u.dot(force);
  };

  const Eigen::VectorXd g =
      2.0 * autodiff::gradient(f, autodiff::wrt(gc_1, gc_2, gc_3), autodiff::at(gc_1, gc_2, gc_3));

  return {g(0), g(1), g(2), g(3), g(4), g(5), g(6), g(7), g(8)};
}