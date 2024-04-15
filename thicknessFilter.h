#pragma once

#include "manifold.h"
#include "mesh.h"

#include <memory>
#include <vector>

#include "eigen3/Eigen/Sparse"
#include <eigen3/Eigen/CholmodSupport>

class ThicknessFilter {

public:
  ThicknessFilter(const Mesh& mesh, const double support_radius)
      : radius_(support_radius / (std::sqrt(3) * 2.0)),
        radius_sq_(radius_ * radius_),
        support_radius_(support_radius),
        mesh_(mesh) {
    update_filter_system();
  };

  void update_filter_system();

  // filter on mesh level
  std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) const;
  std::vector<std::vector<double>> backward(const std::vector<std::vector<double>>& input) const;

  void apply_mass_matrix(const size_t manifold_index, const std::vector<double>& input,
                         std::vector<double>& output) const;

  const double radius_;
  const double radius_sq_;
  const double support_radius_;
  const Mesh& mesh_;

private:
  using SystemMatrix = Eigen::SparseMatrix<double>;
  using Solver       = Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>>;

  void forward_manifold(const uint32_t manifold_index, const std::vector<double>& input,
                        std::vector<double>& result) const;

  void backward_manifold(const uint32_t manifold_index, const std::vector<double>& input,
                         std::vector<double>& result) const;

  std::shared_ptr<SystemMatrix> assemble_system_matrix(const uint32_t manifold_index) const;

  Eigen::Matrix<double, 3, 3> get_element_matrix(const size_t& manifold_index, const size_t face_index) const;
  static std::array<uint32_t, 3> get_element_dof(const Manifold& manifold, const size_t face_index);

  std::vector<std::vector<std::array<double, 3>>> points_at_filter_;
  std::vector<std::shared_ptr<SystemMatrix>> mats_;
  std::vector<std::shared_ptr<Solver>> solvers_;
};
