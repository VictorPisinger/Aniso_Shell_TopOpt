#pragma once

#include <map>
#include <set>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Sparse"
#include <eigen3/Eigen/CholmodSupport>

#include "elementIntegrator.h"
#include "mesh.h"

class FiniteElementAnalysis {

public:
  explicit FiniteElementAnalysis(const Mesh& mesh, const double E = 1.0, const double nu = 0.3,
                                 const std::vector<std::pair<size_t, double>> pressure_manifolds = {})
      : ei(E, nu),
        mesh_(mesh),
        forces(mesh.numberOfDofs),
        stiffnessMatrix(mesh.numberOfDofs, mesh.numberOfDofs),
        pressure_manifolds_(pressure_manifolds) {
    this->forces *= 0.0;
  }

  void set_fixed_dof(std::vector<uint32_t> fd);

  void set_forces(std::vector<std::pair<uint32_t, double>> forceValues);

  void set_prescribed_displacements(std::vector<std::pair<uint32_t, double>> displacementss);

  Eigen::VectorXd solve_system();

  double compute_compliance(const Eigen::VectorXd displacement) const;

  std::vector<std::vector<double>> compute_thickness_gradient(const Eigen::VectorXd displacement) const;

  std::vector<double> compute_shape_gradient(const Eigen::VectorXd& displacement) const;

  std::array<std::vector<std::vector<std::array<double, 6>>>, 2> compute_stress(Eigen::VectorXd displacement) const;

  std::vector<std::vector<double>> compute_vm_stress(Eigen::VectorXd displacement) const;

  std::array<double, 2> compute_compliances(Eigen::VectorXd displacement) const;

  std::array<std::vector<std::vector<double>>, 2> compute_energy(Eigen::VectorXd displacement) const;

  const ElementIntegrator ei;

  const Eigen::VectorXd& Forces() const { return forces; }

  const Mesh& mesh_;

private:
  Eigen::VectorXd forces;
  std::set<uint32_t> fixedDofs;
  std::map<uint32_t, double> prescribedDofs;

  Eigen::SparseMatrix<double> stiffnessMatrix;
  Eigen::CholmodDecomposition<Eigen::SparseMatrix<double>> solver;
  bool solver_sparsity_is_set = false;

  const std::vector<std::pair<size_t, double>> pressure_manifolds_;

  void add_pressure_loads(Eigen::VectorXd& forces) const;

  void add_pressure_shape_gradients(const Eigen::VectorXd& displacement, std::vector<double>& gradient) const;

  void assemble_stiffness_matrix(Eigen::SparseMatrix<double>& mat) const;

  void apply_boundary_conditions(Eigen::SparseMatrix<double>& mat, Eigen::VectorXd& force) const;

  Eigen::Matrix<double, 12, 12> compute_local_stiffness(const size_t manifold_index, const size_t face_index) const;

  std::array<uint32_t, 12> compute_local_dof(const size_t manifold_index, const size_t face_index) const;

  static std::array<bool, 3> compute_rotation_flips(std::array<uint32_t, 3> nodeIndices);
};