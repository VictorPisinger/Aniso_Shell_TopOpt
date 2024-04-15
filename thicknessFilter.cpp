#include "thicknessFilter.h"

#include "pdeFilter.h"

#include "shellElement.h"

#include <iostream>

void ThicknessFilter::update_filter_system() {

  points_at_filter_.resize(mesh_.manifolds.size());
  for (size_t i = 0; i < mesh_.manifolds.size(); i++) {
    points_at_filter_[i] = mesh_.manifolds[i].Points();
  }

  mats_.resize(mesh_.manifolds.size());
  solvers_.resize(mesh_.manifolds.size());

  for (size_t i = 0; i < mesh_.manifolds.size(); i++) {
    mats_[i]    = assemble_system_matrix(i);
    solvers_[i] = std::make_shared<Solver>();
    solvers_[i]->compute(*mats_[i]);
  }
}

std::vector<std::vector<double>> ThicknessFilter::forward(const std::vector<std::vector<double>>& input) const {

  std::vector<std::vector<double>> result(input.size());

  for (size_t mi = 0; mi < mesh_.manifolds.size(); mi++) {
    result[mi].resize(input[mi].size(), 0.0);
    forward_manifold(mi, input[mi], result[mi]);
  }

  return result;
}

std::vector<std::vector<double>> ThicknessFilter::backward(const std::vector<std::vector<double>>& input) const {

  std::vector<std::vector<double>> result(input.size());
  for (size_t mi = 0; mi < mesh_.manifolds.size(); mi++) {
    result[mi].resize(input[mi].size(), 0.0);
    backward_manifold(mi, input[mi], result[mi]);
  }

  return result;
}

void ThicknessFilter::forward_manifold(const uint32_t manifold_index, const std::vector<double>& input,
                                       std::vector<double>& result) const {

  std::vector<double> tmp(input.size(), 0.0);
  apply_mass_matrix(manifold_index, input, tmp);

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    rhs(i) = tmp[i];
  }

  Eigen::VectorXd solution = solvers_[manifold_index]->solve(rhs);

  for (size_t i = 0; i < input.size(); i++) {
    result[i] = solution(i);
  }
}

void ThicknessFilter::backward_manifold(const uint32_t manifold_index, const std::vector<double>& input,
                                        std::vector<double>& result) const {

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    rhs(i) = input[i];
  }

  Eigen::VectorXd solution = solvers_[manifold_index]->solve(rhs);

  std::vector<double> tmp(input.size(), 0.0);
  for (size_t i = 0; i < input.size(); i++) {
    tmp[i] = solution(i);
  }

  apply_mass_matrix(manifold_index, tmp, result);
}

Eigen::Matrix<double, 3, 3> ThicknessFilter::get_element_matrix(const size_t& manifold_index,
                                                                const size_t face_index) const {

  const auto& mi       = manifold_index;
  const auto& manifold = mesh_.manifolds[mi];
  const auto& f        = manifold.Faces()[face_index];

  std::array<Eigen::Vector3d, 3> coordinates;
  coordinates[0] = {points_at_filter_[mi][f[0]][0], points_at_filter_[mi][f[0]][1], points_at_filter_[mi][f[0]][2]};
  coordinates[1] = {points_at_filter_[mi][f[1]][0], points_at_filter_[mi][f[1]][1], points_at_filter_[mi][f[1]][2]};
  coordinates[2] = {points_at_filter_[mi][f[2]][0], points_at_filter_[mi][f[2]][1], points_at_filter_[mi][f[2]][2]};

  const auto m = Shell::compute_geometry_mass(coordinates);
  const auto k = Shell::compute_geometry_laplace(coordinates);

  return (radius_sq_ * k) + m;
}

std::array<uint32_t, 3> ThicknessFilter::get_element_dof(const Manifold& manifold, const size_t face_index) {

  const auto& f = manifold.Faces()[face_index];

  const uint32_t node1 = f[0];
  const uint32_t node2 = f[1];
  const uint32_t node3 = f[2];

  return {node1, node2, node3};
}

std::shared_ptr<ThicknessFilter::SystemMatrix>
ThicknessFilter::assemble_system_matrix(const uint32_t manifold_index) const {

  const auto& manifold        = mesh_.manifolds[manifold_index];
  const auto number_of_points = manifold.Points().size();

  std::vector<Eigen::Triplet<double>> t;
  t.reserve(3 * 3 * number_of_points);

  // #pragma omp parallel for schedule(static) default(none) shared(t) firstprivate(manifold)
  for (size_t i = 0; i < manifold.Faces().size(); i++) {

    const auto a    = get_element_matrix(manifold_index, i);
    const auto dofs = get_element_dof(manifold, i);

    // loop column first, as eigen matrices are stored column-major
    // #pragma omp critical
    {
      for (int jj = 0; jj < 3; jj++) {
        for (int ii = 0; ii < 3; ii++) {
          t.emplace_back(dofs[ii], dofs[jj], a(ii, jj));
        }
      }
    }
  }

  std::shared_ptr<SystemMatrix> mat_ptr = std::make_shared<SystemMatrix>(number_of_points, number_of_points);

  mat_ptr->setFromTriplets(t.begin(), t.end());
  mat_ptr->makeCompressed();

  return mat_ptr;
}

void ThicknessFilter::apply_mass_matrix(const size_t manifold_index, const std::vector<double>& input,
                                        std::vector<double>& output) const {

  const auto mi        = manifold_index;
  const auto& manifold = mesh_.manifolds[manifold_index];

  for (size_t fi = 0; fi < manifold.Faces().size(); fi++) {
    const auto& f = manifold.Faces()[fi];

    std::array<Eigen::Vector3d, 3> coordinates;
    coordinates[0] = {points_at_filter_[mi][f[0]][0], points_at_filter_[mi][f[0]][1], points_at_filter_[mi][f[0]][2]};
    coordinates[1] = {points_at_filter_[mi][f[1]][0], points_at_filter_[mi][f[1]][1], points_at_filter_[mi][f[1]][2]};
    coordinates[2] = {points_at_filter_[mi][f[2]][0], points_at_filter_[mi][f[2]][1], points_at_filter_[mi][f[2]][2]};

    const auto dofs = get_element_dof(manifold, fi);
    const auto m    = Shell::compute_geometry_mass(coordinates);

    // read values
    Eigen::Vector3d values = {input[dofs[0]], input[dofs[1]], input[dofs[2]]};

    // apply x = M x
    values = m * values;

    // write back result
    output[dofs[0]] += values[0];
    output[dofs[1]] += values[1];
    output[dofs[2]] += values[2];
  }
}
