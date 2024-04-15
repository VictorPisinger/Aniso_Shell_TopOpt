#pragma once

#include "manifold.h"
#include "mesh.h"

#include <memory>
#include <vector>

#include "eigen3/Eigen/Sparse"
#include <eigen3/Eigen/CholmodSupport>

class PdeFilter {

public:
  PdeFilter(const Mesh& mesh, const double support_radius,
            const std::array<std::shared_ptr<std::vector<uint32_t>>, 3> boundary_nodes = {nullptr, nullptr, nullptr})
      : radius_(support_radius / (std::sqrt(3) * 2.0)),
        radius_sq_(radius_ * radius_),
        support_radius_(support_radius),
        mesh_(mesh),
        boundary_nodes_(boundary_nodes) {
    update_filter_system();
  };

  void update_filter_system();

  // filter on mesh level
  std::vector<double> forward(const std::vector<double>& input) const;
  std::vector<double> backward(std::vector<double> input) const;

  void apply_mass_matrix(const std::vector<double>& input, std::vector<double>& output) const;

  const double radius_;
  const double radius_sq_;
  const double support_radius_;
  const Mesh& mesh_;
  const std::array<std::shared_ptr<std::vector<uint32_t>>, 3> boundary_nodes_;

private:
  using SystemMatrix = Eigen::SparseMatrix<double>;
  using Solver       = Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>>;

  std::shared_ptr<SystemMatrix>
  assemble_system_matrix(const std::shared_ptr<std::vector<uint32_t>> boundary_conditions) const;

  Eigen::Matrix<double, 3, 3> get_element_matrix(const size_t& manifold_index, const size_t face_index) const;

  static std::array<uint32_t, 3> get_element_scalar_dof(const Manifold& manifold, const size_t face_index);
  static std::array<uint32_t, 9> get_element_dof(const Manifold& manifold, const size_t face_index);

  void scatter_to_scalar(const std::vector<double>& input, Eigen::VectorXd& output, const size_t offset) const;
  void scatter_from_scalar(const Eigen::VectorXd& input, std::vector<double>& output, const size_t offset) const;

  std::vector<std::vector<std::array<double, 3>>> points_at_filter_;

  std::array<std::shared_ptr<SystemMatrix>, 3> mats_;
  std::array<std::shared_ptr<Solver>, 3> solvers_;
};
