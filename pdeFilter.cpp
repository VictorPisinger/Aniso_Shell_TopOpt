#include "pdeFilter.h"

#include "shellElement.h"

#include <iostream>

void PdeFilter::update_filter_system() {

  points_at_filter_.resize(mesh_.manifolds.size());
  for (size_t i = 0; i < mesh_.manifolds.size(); i++) {
    points_at_filter_[i] = mesh_.manifolds[i].Points();
  }

  // always setup x matrix
  mats_[0]    = assemble_system_matrix(boundary_nodes_[0]);
  solvers_[0] = std::make_shared<Solver>();
  solvers_[0]->compute(*mats_[0]);

  // if bc_x == bc_y copy reference for solver, else setup
  if (boundary_nodes_[1] == boundary_nodes_[0]) {
    mats_[1]    = mats_[0];
    solvers_[1] = solvers_[0];
  } else {
    mats_[1]    = assemble_system_matrix(boundary_nodes_[1]);
    solvers_[1] = std::make_shared<Solver>();
    solvers_[1]->compute(*mats_[1]);
  }

  // check if copy is possible for z
  if (boundary_nodes_[2] == boundary_nodes_[0]) {
    mats_[2]    = mats_[0];
    solvers_[2] = solvers_[0];
  } else if (boundary_nodes_[2] == boundary_nodes_[1]) {
    mats_[2]    = mats_[1];
    solvers_[2] = solvers_[1];
  } else {
    mats_[2]    = assemble_system_matrix(boundary_nodes_[2]);
    solvers_[2] = std::make_shared<Solver>();
    solvers_[2]->compute(*mats_[2]);
  }
}

std::vector<double> PdeFilter::forward(const std::vector<double>& input) const {

  std::vector<double> tmp(input.size(), 0.0);
  std::vector<double> result(input.size(), 0.0);

  apply_mass_matrix(input, tmp);

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(mesh_.numberOfUniqueNodes);

  scatter_to_scalar(tmp, rhs, 0);
  Eigen::VectorXd solution = solvers_[0]->solve(rhs);
  scatter_from_scalar(solution, result, 0);

  scatter_to_scalar(tmp, rhs, 1);
  solution = solvers_[1]->solve(rhs);
  scatter_from_scalar(solution, result, 1);

  scatter_to_scalar(tmp, rhs, 2);
  solution = solvers_[2]->solve(rhs);
  scatter_from_scalar(solution, result, 2);

  return result;
}

std::vector<double> PdeFilter::backward(std::vector<double> input) const {

  std::vector<double> tmp(input.size(), 0.0);
  std::vector<double> result(input.size(), 0.0);

  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(mesh_.numberOfUniqueNodes);

  scatter_to_scalar(input, rhs, 0);
  Eigen::VectorXd solution = solvers_[0]->solve(rhs);
  scatter_from_scalar(solution, tmp, 0);

  scatter_to_scalar(input, rhs, 1);
  solution = solvers_[1]->solve(rhs);
  scatter_from_scalar(solution, tmp, 1);

  scatter_to_scalar(input, rhs, 2);
  solution = solvers_[2]->solve(rhs);
  scatter_from_scalar(solution, tmp, 2);

  apply_mass_matrix(tmp, result);
  return result;
}

void PdeFilter::scatter_to_scalar(const std::vector<double>& input, Eigen::VectorXd& output,
                                  const size_t offset) const {

  for (size_t i = 0; i < mesh_.numberOfUniqueNodes; i++) {
    output(i) = input[3 * i + offset];
  }

  if (boundary_nodes_[offset] != nullptr)
    for (const auto& fd : *boundary_nodes_[offset]) {
      output(fd) = 0.0;
    }
}

void PdeFilter::scatter_from_scalar(const Eigen::VectorXd& input, std::vector<double>& output,
                                    const size_t offset) const {

  for (size_t i = 0; i < mesh_.numberOfUniqueNodes; i++) {
    output[3 * i + offset] = input(i);
  }
}

Eigen::Matrix<double, 3, 3> PdeFilter::get_element_matrix(const size_t& manifold_index, const size_t face_index) const {

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

std::array<uint32_t, 3> PdeFilter::get_element_scalar_dof(const Manifold& manifold, const size_t face_index) {

  const auto& f        = manifold.Faces()[face_index];
  const uint32_t node1 = manifold.UniqueNodeNumber()[f[0]];
  const uint32_t node2 = manifold.UniqueNodeNumber()[f[1]];
  const uint32_t node3 = manifold.UniqueNodeNumber()[f[2]];

  return {node1, node2, node3};
}

std::array<uint32_t, 9> PdeFilter::get_element_dof(const Manifold& manifold, const size_t face_index) {

  std::array<uint32_t, 9> dofs;

  const auto& f = manifold.Faces()[face_index];

  const uint32_t node1 = manifold.UniqueNodeNumber()[f[0]];
  const uint32_t node2 = manifold.UniqueNodeNumber()[f[1]];
  const uint32_t node3 = manifold.UniqueNodeNumber()[f[2]];

  // u
  dofs[0] = 3 * node1 + 0;
  dofs[1] = 3 * node2 + 0;
  dofs[2] = 3 * node3 + 0;
  dofs[3] = 3 * node1 + 1;
  dofs[4] = 3 * node2 + 1;
  dofs[5] = 3 * node3 + 1;
  dofs[6] = 3 * node1 + 2;
  dofs[7] = 3 * node2 + 2;
  dofs[8] = 3 * node3 + 2;

  return dofs;
}

std::shared_ptr<PdeFilter::SystemMatrix>
PdeFilter::assemble_system_matrix(const std::shared_ptr<std::vector<uint32_t>> boundary_conditions) const {

  std::vector<Eigen::Triplet<double>> t;
  t.reserve(3 * 3 * mesh_.number_of_triangles());

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const auto& manifold = mesh_.manifolds[m];

    // #pragma omp parallel for schedule(static) default(none) shared(t) firstprivate(manifold)
    for (size_t i = 0; i < manifold.Faces().size(); i++) {

      const auto a    = get_element_matrix(m, i);
      const auto dofs = get_element_scalar_dof(manifold, i);

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
  }

  std::shared_ptr<SystemMatrix> mat_ptr =
      std::make_shared<SystemMatrix>(mesh_.numberOfUniqueNodes, mesh_.numberOfUniqueNodes);

  mat_ptr->setFromTriplets(t.begin(), t.end());

  Eigen::VectorXd nullMatrix  = Eigen::VectorXd::Ones(mesh_.numberOfUniqueNodes);
  Eigen::VectorXd inverseNull = Eigen::VectorXd::Zero(mesh_.numberOfUniqueNodes);

  if (boundary_conditions != nullptr) {
    for (const auto& fd : *boundary_conditions) {
      nullMatrix.coeffRef(fd)  = 0.0;
      inverseNull.coeffRef(fd) = 1.0;
    }
  }

  *mat_ptr = (nullMatrix.asDiagonal() * (*mat_ptr) * nullMatrix.asDiagonal());
  *mat_ptr += inverseNull.asDiagonal();

  mat_ptr->makeCompressed();

  return mat_ptr;
}

void PdeFilter::apply_mass_matrix(const std::vector<double>& input, std::vector<double>& output) const {

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const auto& manifold = mesh_.manifolds[m];

    for (size_t i = 0; i < manifold.Faces().size(); i++) {

      const auto f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {points_at_filter_[m][f[0]][0], points_at_filter_[m][f[0]][1], points_at_filter_[m][f[0]][2]};
      coordinates[1] = {points_at_filter_[m][f[1]][0], points_at_filter_[m][f[1]][1], points_at_filter_[m][f[1]][2]};
      coordinates[2] = {points_at_filter_[m][f[2]][0], points_at_filter_[m][f[2]][1], points_at_filter_[m][f[2]][2]};
      const auto m   = Shell::compute_geometry_mass(coordinates);

      const auto dofs = get_element_dof(manifold, i);

      // read values
      Eigen::Vector3d xValues = {input[dofs[0]], input[dofs[1]], input[dofs[2]]};
      Eigen::Vector3d yValues = {input[dofs[3]], input[dofs[4]], input[dofs[5]]};
      Eigen::Vector3d zValues = {input[dofs[6]], input[dofs[7]], input[dofs[8]]};

      // apply x = M x
      xValues = m * xValues;
      yValues = m * yValues;
      zValues = m * zValues;

      // write back result
      output[dofs[0]] += xValues[0];
      output[dofs[1]] += xValues[1];
      output[dofs[2]] += xValues[2];
      output[dofs[3]] += yValues[0];
      output[dofs[4]] += yValues[1];
      output[dofs[5]] += yValues[2];
      output[dofs[6]] += zValues[0];
      output[dofs[7]] += zValues[1];
      output[dofs[8]] += zValues[2];
    }
  }
}