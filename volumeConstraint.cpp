#include "volumeConstraint.h"

#include <array>

#include "shellElement.h"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

double VolumeConstraint::compute_volume(const Manifold& manifold) {
  double volume = 0.0;
  for (size_t i = 0; i < manifold.Faces().size(); i++) {

    const size_t idx1 = manifold.Faces()[i][0];
    const size_t idx2 = manifold.Faces()[i][1];
    const size_t idx3 = manifold.Faces()[i][2];

    const Eigen::Vector3d p1 = {manifold.Points()[idx1][0], manifold.Points()[idx1][1], manifold.Points()[idx1][2]};
    const Eigen::Vector3d p2 = {manifold.Points()[idx2][0], manifold.Points()[idx2][1], manifold.Points()[idx2][2]};
    const Eigen::Vector3d p3 = {manifold.Points()[idx3][0], manifold.Points()[idx3][1], manifold.Points()[idx3][2]};

    const double thickness =
        (manifold.Thickness()[idx1] + manifold.Thickness()[idx2] + manifold.Thickness()[idx3]) / 3.0;

    volume += Shell::compute_volume<double>({p1, p2, p3}, thickness);
  }

  return volume;
}

double VolumeConstraint::compute_volume(const Mesh& mesh) {
  double volume = 0.0;
  for (const auto& manifold : mesh.manifolds) {
    volume += compute_volume(manifold);
  }
  return volume;
}

double VolumeConstraint::compute_value() const {
  double volume = compute_volume(mesh_);

  return (volume / allowedVolume_) - 1.0;
}

std::vector<std::vector<double>> VolumeConstraint::compute_thickness_gradient() const {

  std::vector<std::vector<double>> gradients;
  gradients.resize(mesh_.manifolds.size());

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const auto& manifold = mesh_.manifolds[m];
    gradients[m].resize(manifold.Points().size(), 0.0);

    for (size_t i = 0; i < manifold.Faces().size(); i++) {

      const size_t idx1 = manifold.Faces()[i][0];
      const size_t idx2 = manifold.Faces()[i][1];
      const size_t idx3 = manifold.Faces()[i][2];

      const Eigen::Vector3d p1 = {manifold.Points()[idx1][0], manifold.Points()[idx1][1], manifold.Points()[idx1][2]};
      const Eigen::Vector3d p2 = {manifold.Points()[idx2][0], manifold.Points()[idx2][1], manifold.Points()[idx2][2]};
      const Eigen::Vector3d p3 = {manifold.Points()[idx3][0], manifold.Points()[idx3][1], manifold.Points()[idx3][2]};
      const double area        = Shell::compute_area<double>({p1, p2, p3});

      const double pointGradient = (1.0 / 3.0) * (1.0 / allowedVolume_) * area;

      gradients[m][idx1] += pointGradient;
      gradients[m][idx2] += pointGradient;
      gradients[m][idx3] += pointGradient;
    }
  }

  return gradients;
}

std::vector<double> VolumeConstraint::compute_shape_gradient() const {

  std::vector<double> gradients(3 * mesh_.numberOfUniqueNodes, 0.0);

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const auto& manifold = mesh_.manifolds[m];

    for (size_t i = 0; i < manifold.Faces().size(); i++) {

      const size_t idx1 = manifold.Faces()[i][0];
      const size_t idx2 = manifold.Faces()[i][1];
      const size_t idx3 = manifold.Faces()[i][2];

      Eigen::Vector<autodiff::real, 3> p1 = {manifold.Points()[idx1][0], manifold.Points()[idx1][1],
                                             manifold.Points()[idx1][2]};
      Eigen::Vector<autodiff::real, 3> p2 = {manifold.Points()[idx2][0], manifold.Points()[idx2][1],
                                             manifold.Points()[idx2][2]};
      Eigen::Vector<autodiff::real, 3> p3 = {manifold.Points()[idx3][0], manifold.Points()[idx3][1],
                                             manifold.Points()[idx3][2]};

      autodiff::real thickness =
          (manifold.Thickness()[idx1] + manifold.Thickness()[idx2] + manifold.Thickness()[idx3]) / 3.0;

      const auto f = [](const Eigen::Vector<autodiff::real, 3>& pp1, const Eigen::Vector<autodiff::real, 3>& pp2,
                        const Eigen::Vector<autodiff::real, 3>& pp3, const autodiff::real& thk) {
        const std::array<Eigen::Vector<autodiff::real, 3>, 3> pp = {pp1, pp2, pp3};
        return Shell::compute_volume<autodiff::real>(pp, thk);
      };

      const size_t n1 = manifold.UniqueNodeNumber()[idx1];
      const size_t n2 = manifold.UniqueNodeNumber()[idx2];
      const size_t n3 = manifold.UniqueNodeNumber()[idx3];

      gradients[3 * n1 + 0] += autodiff::derivative(f, autodiff::wrt(p1(0)), autodiff::at(p1, p2, p3, thickness));
      gradients[3 * n1 + 1] += autodiff::derivative(f, autodiff::wrt(p1(1)), autodiff::at(p1, p2, p3, thickness));
      gradients[3 * n1 + 2] += autodiff::derivative(f, autodiff::wrt(p1(2)), autodiff::at(p1, p2, p3, thickness));

      gradients[3 * n2 + 0] += autodiff::derivative(f, autodiff::wrt(p2(0)), autodiff::at(p1, p2, p3, thickness));
      gradients[3 * n2 + 1] += autodiff::derivative(f, autodiff::wrt(p2(1)), autodiff::at(p1, p2, p3, thickness));
      gradients[3 * n2 + 2] += autodiff::derivative(f, autodiff::wrt(p2(2)), autodiff::at(p1, p2, p3, thickness));

      gradients[3 * n3 + 0] += autodiff::derivative(f, autodiff::wrt(p3(0)), autodiff::at(p1, p2, p3, thickness));
      gradients[3 * n3 + 1] += autodiff::derivative(f, autodiff::wrt(p3(1)), autodiff::at(p1, p2, p3, thickness));
      gradients[3 * n3 + 2] += autodiff::derivative(f, autodiff::wrt(p3(2)), autodiff::at(p1, p2, p3, thickness));
    }
  }

  // scale to constraint range
  const double scale = 1.0 / allowedVolume_;
  for (auto& v : gradients)
    v *= scale;

  return gradients;
}