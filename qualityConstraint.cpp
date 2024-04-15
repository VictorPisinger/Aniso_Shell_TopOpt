#include "qualityConstraint.h"

#include "shellElement.h"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

double QualityConstraint::compute_AR_true() const {

  double max_ar = 1.0;

  for (const auto& manifold : mesh_.manifolds) {
    for (size_t i = 0; i < manifold.Faces().size(); i++) {
      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      max_ar = std::max(max_ar, Shell::compute_aspect_ratio(coordinates));
    }
  }

  return max_ar;
}

double QualityConstraint::compute_AR_value() const {

  // sum the aggregated value
  double sum                 = 0.0;
  double number_of_triangles = 0.0;

  for (const auto& manifold : mesh_.manifolds) {
    for (size_t i = 0; i < manifold.Faces().size(); i++) {
      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      double aspect_ratio = Shell::compute_aspect_ratio(coordinates);
      sum += std::pow(aspect_ratio, p_);
      number_of_triangles += 1.0;
    }
  }

  const double scale      = 1.0 / number_of_triangles;
  const double constraint = std::pow(scale * sum, 1.0 / p_) - allowedAspectRatio_;

  return constraint;
}

std::vector<double> QualityConstraint::compute_AR_shape_gradient() const {

  std::vector<double> gradients(3 * mesh_.numberOfUniqueNodes, 0.0);

  // sum the aggregated value
  double sum                 = 0.0;
  double number_of_triangles = 0.0;

  for (const auto& manifold : mesh_.manifolds) {
    for (size_t i = 0; i < manifold.Faces().size(); i++) {
      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      double aspect_ratio = Shell::compute_aspect_ratio(coordinates);
      sum += std::pow(aspect_ratio, p_);
      number_of_triangles += 1.0;
    }
  }

  const double scale = 1.0 / number_of_triangles;
  const double fact  = scale * std::pow(scale * sum, (1.0 / p_) - 1.0);

  const auto fun = [](const Eigen::Vector<autodiff::real, 3>& pp1, const Eigen::Vector<autodiff::real, 3>& pp2,
                      const Eigen::Vector<autodiff::real, 3>& pp3) {
    const std::array<Eigen::Vector<autodiff::real, 3>, 3> pp = {pp1, pp2, pp3};
    return Shell::compute_aspect_ratio<autodiff::real>(pp);
  };

  for (const auto& manifold : mesh_.manifolds) {
    for (size_t i = 0; i < manifold.Faces().size(); i++) {
      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const double aspect_ratio = Shell::compute_aspect_ratio(coordinates);
      const double dAR          = fact * std::pow(aspect_ratio, p_ - 1.0);

      const size_t n1 = manifold.UniqueNodeNumber()[f[0]];
      const size_t n2 = manifold.UniqueNodeNumber()[f[1]];
      const size_t n3 = manifold.UniqueNodeNumber()[f[2]];

      Eigen::Vector<autodiff::real, 3> p1 = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1],
                                             manifold.Points()[f[0]][2]};
      Eigen::Vector<autodiff::real, 3> p2 = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1],
                                             manifold.Points()[f[1]][2]};
      Eigen::Vector<autodiff::real, 3> p3 = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1],
                                             manifold.Points()[f[2]][2]};

      gradients[3 * n1 + 0] += dAR * autodiff::derivative(fun, autodiff::wrt(p1(0)), autodiff::at(p1, p2, p3));
      gradients[3 * n1 + 1] += dAR * autodiff::derivative(fun, autodiff::wrt(p1(1)), autodiff::at(p1, p2, p3));
      gradients[3 * n1 + 2] += dAR * autodiff::derivative(fun, autodiff::wrt(p1(2)), autodiff::at(p1, p2, p3));

      gradients[3 * n2 + 0] += dAR * autodiff::derivative(fun, autodiff::wrt(p2(0)), autodiff::at(p1, p2, p3));
      gradients[3 * n2 + 1] += dAR * autodiff::derivative(fun, autodiff::wrt(p2(1)), autodiff::at(p1, p2, p3));
      gradients[3 * n2 + 2] += dAR * autodiff::derivative(fun, autodiff::wrt(p2(2)), autodiff::at(p1, p2, p3));

      gradients[3 * n3 + 0] += dAR * autodiff::derivative(fun, autodiff::wrt(p3(0)), autodiff::at(p1, p2, p3));
      gradients[3 * n3 + 1] += dAR * autodiff::derivative(fun, autodiff::wrt(p3(1)), autodiff::at(p1, p2, p3));
      gradients[3 * n3 + 2] += dAR * autodiff::derivative(fun, autodiff::wrt(p3(2)), autodiff::at(p1, p2, p3));
    }
  }

  return gradients;
}
