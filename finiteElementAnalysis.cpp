#include "finiteElementAnalysis.h"

#include <eigen3/Eigen/CholmodSupport>
#include <eigen3/Eigen/UmfPackSupport>

#include <algorithm>
#include <iostream>

void FiniteElementAnalysis::set_fixed_dof(std::vector<uint32_t> fd) {
  this->fixedDofs.clear();
  this->fixedDofs.insert(fd.begin(), fd.end());
}

void FiniteElementAnalysis::set_forces(std::vector<std::pair<uint32_t, double>> forceValues) {
  for (const auto& v : forceValues) {
    forces(v.first) += v.second;
  }
}

void FiniteElementAnalysis::set_prescribed_displacements(std::vector<std::pair<uint32_t, double>> displacements) {
  this->prescribedDofs.clear();
  this->prescribedDofs.insert(displacements.begin(), displacements.end());
}

Eigen::VectorXd FiniteElementAnalysis::solve_system() {
  Eigen::VectorXd displacements(mesh_.numberOfDofs);
  Eigen::VectorXd current_forces = this->forces;

  if (!this->pressure_manifolds_.empty()) {
    add_pressure_loads(current_forces);
  }

  assemble_stiffness_matrix(stiffnessMatrix);
  apply_boundary_conditions(stiffnessMatrix, current_forces);

  if (!solver_sparsity_is_set) {
    solver.analyzePattern(stiffnessMatrix);
    solver_sparsity_is_set = true;
  }
  solver.factorize(stiffnessMatrix);

  if (solver.info() != Eigen::Success) {
    std::cout << "FEA MATRIX FACTORIZATION FAILED! due to ";

    if (solver.info() == Eigen::NumericalIssue)
      std::cout << "Nummerical issues";

    if (solver.info() == Eigen::InvalidInput)
      std::cout << "Invalid input";

    return displacements;
  }

  displacements = solver.solve(current_forces);

  return displacements;
}

std::array<std::vector<std::vector<std::array<double, 6>>>, 2>
FiniteElementAnalysis::compute_stress(Eigen::VectorXd displacement) const {

  std::vector<std::vector<std::array<double, 6>>> membrane(mesh_.manifolds.size());
  std::vector<std::vector<std::array<double, 6>>> bending(mesh_.manifolds.size());

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const size_t num_faces = mesh_.manifolds[m].Faces().size();
    membrane[m].resize(num_faces);
    bending[m].resize(num_faces);

    for (size_t i = 0; i < num_faces; i++) {

      const auto& manifold = mesh_.manifolds[m];
      const auto& f        = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto& thk        = manifold.Thickness();
      const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

      const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
      const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
      const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
      const auto flipRotations = compute_rotation_flips({node1, node2, node3});

      const auto dofs = compute_local_dof(m, i);
      Eigen::Vector<double, 12> u_element;
      for (int i = 0; i < 12; i++)
        u_element(i) = displacement(dofs[i]);

      const auto stress = ei.compute_element_stress(coordinates, flipRotations, thickness, u_element);
      membrane[m][i]    = stress.first;
      bending[m][i]     = stress.second;
    }
  }

  return {membrane, bending};
}

std::vector<std::vector<double>> FiniteElementAnalysis::compute_vm_stress(Eigen::VectorXd displacement) const {

  std::vector<std::vector<double>> vm_stress(mesh_.manifolds.size());

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

      const auto& manifold = mesh_.manifolds[m];
      const auto& f        = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto& thk        = manifold.Thickness();
      const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

      const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
      const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
      const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
      const auto flipRotations = compute_rotation_flips({node1, node2, node3});

      const auto dofs = compute_local_dof(m, i);
      Eigen::Vector<double, 12> u_element;
      for (int i = 0; i < 12; i++)
        u_element(i) = displacement(dofs[i]);

      const auto stress = ei.compute_vm_stress(coordinates, flipRotations, thickness, u_element);
      vm_stress[m].push_back(stress);
    }
  }

  return vm_stress;
}

std::array<std::vector<std::vector<double>>, 2>
FiniteElementAnalysis::compute_energy(Eigen::VectorXd displacement) const {

  std::vector<std::vector<double>> membrane(mesh_.manifolds.size());
  std::vector<std::vector<double>> bending(mesh_.manifolds.size());

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

      const auto& manifold = mesh_.manifolds[m];
      const auto& f        = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto& thk        = manifold.Thickness();
      const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

      const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
      const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
      const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
      const auto flipRotations = compute_rotation_flips({node1, node2, node3});

      const auto dofs = compute_local_dof(m, i);
      Eigen::Vector<double, 12> u_element;
      for (int i = 0; i < 12; i++)
        u_element(i) = displacement(dofs[i]);

      const auto energy = ei.compute_bending_and_membrane_compliance(coordinates, flipRotations, thickness, u_element);
      membrane[m].push_back(energy[0]);
      bending[m].push_back(energy[1]);
    }
  }

  return {membrane, bending};
}

std::array<double, 2> FiniteElementAnalysis::compute_compliances(Eigen::VectorXd displacement) const {

  double membrane = 0.0;
  double bending  = 0.0;

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

      const auto& manifold = mesh_.manifolds[m];
      const auto& f        = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto& thk        = manifold.Thickness();
      const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

      const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
      const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
      const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
      const auto flipRotations = compute_rotation_flips({node1, node2, node3});

      const auto dofs = compute_local_dof(m, i);
      Eigen::Vector<double, 12> u_element;
      for (int i = 0; i < 12; i++)
        u_element(i) = displacement(dofs[i]);

      const auto energy = ei.compute_bending_and_membrane_compliance(coordinates, flipRotations, thickness, u_element);
      membrane += energy[0];
      bending += energy[1];
    }
  }
  return {membrane, bending};
}

double FiniteElementAnalysis::compute_compliance(const Eigen::VectorXd displacement) const {

  Eigen::VectorXd current_forces = this->forces;

  if (!this->pressure_manifolds_.empty()) {
    add_pressure_loads(current_forces);
  }

  return displacement.transpose() * current_forces;
}

std::vector<std::vector<double>>
FiniteElementAnalysis::compute_thickness_gradient(const Eigen::VectorXd displacement) const {

  std::vector<std::vector<double>> gradient(mesh_.manifolds.size());

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    gradient[m].resize(mesh_.manifolds[m].Points().size(), 0.0);

    for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

      const auto& manifold = mesh_.manifolds[m];
      const auto& f        = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto& thk        = manifold.Thickness();
      const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

      const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
      const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
      const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
      const auto flipRotations = compute_rotation_flips({node1, node2, node3});

      const auto dofs = compute_local_dof(m, i);
      Eigen::Vector<double, 12> u_element;
      for (int i = 0; i < 12; i++)
        u_element(i) = displacement(dofs[i]);

      const auto dt = ei.compute_thickness_gradient(coordinates, flipRotations, thickness, u_element);

      gradient[m][f[0]] += dt / 3.0;
      gradient[m][f[1]] += dt / 3.0;
      gradient[m][f[2]] += dt / 3.0;
    }
  }

  return gradient;
}

std::vector<double> FiniteElementAnalysis::compute_shape_gradient(const Eigen::VectorXd& displacement) const {

  std::vector<double> gradient(3 * mesh_.numberOfUniqueNodes, 0.0);

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const auto& manifold = mesh_.manifolds[m];

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < manifold.Faces().size(); i++) {
      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto& thk        = manifold.Thickness();
      const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

      const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
      const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
      const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
      const auto flipRotations = compute_rotation_flips({node1, node2, node3});

      const auto dofs = compute_local_dof(m, i);
      Eigen::Vector<double, 12> u_element;
      for (int i = 0; i < 12; i++)
        u_element(i) = displacement(dofs[i]);

      const auto grad = ei.compute_shape_gradients(coordinates, flipRotations, thickness, u_element);

      const int ni1 = manifold.UniqueNodeNumber()[f[0]];
      const int ni2 = manifold.UniqueNodeNumber()[f[1]];
      const int ni3 = manifold.UniqueNodeNumber()[f[2]];

#pragma omp critical
      {
        gradient[3 * ni1 + 0] += grad[0];
        gradient[3 * ni1 + 1] += grad[1];
        gradient[3 * ni1 + 2] += grad[2];
        gradient[3 * ni2 + 0] += grad[3];
        gradient[3 * ni2 + 1] += grad[4];
        gradient[3 * ni2 + 2] += grad[5];
        gradient[3 * ni3 + 0] += grad[6];
        gradient[3 * ni3 + 1] += grad[7];
        gradient[3 * ni3 + 2] += grad[8];
      }
    }
  }

  if (!this->pressure_manifolds_.empty()) {
    add_pressure_shape_gradients(displacement, gradient);
  }

  return gradient;
}

Eigen::Matrix<double, 12, 12> FiniteElementAnalysis::compute_local_stiffness(const size_t manifold_index,
                                                                             const size_t face_index) const {

  const auto& manifold = mesh_.manifolds[manifold_index];
  const auto& f        = manifold.Faces()[face_index];

  std::array<Eigen::Vector3d, 3> coordinates;
  coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
  coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
  coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

  const auto& thk        = manifold.Thickness();
  const double thickness = (thk[f[0]] + thk[f[1]] + thk[f[2]]) / 3.0;

  const uint32_t node1     = manifold.UniqueNodeNumber()[f[0]];
  const uint32_t node2     = manifold.UniqueNodeNumber()[f[1]];
  const uint32_t node3     = manifold.UniqueNodeNumber()[f[2]];
  const auto flipRotations = compute_rotation_flips({node1, node2, node3});

  return ei.compute_element_stiffness(coordinates, flipRotations, thickness);
}

std::array<uint32_t, 12> FiniteElementAnalysis::compute_local_dof(const size_t manifold_index,
                                                                  const size_t face_index) const {

  std::array<uint32_t, 12> dofs;

  const auto& manifold = mesh_.manifolds[manifold_index];
  const auto& f        = manifold.Faces()[face_index];

  const uint32_t node1 = manifold.UniqueNodeNumber()[f[0]];
  const uint32_t node2 = manifold.UniqueNodeNumber()[f[1]];
  const uint32_t node3 = manifold.UniqueNodeNumber()[f[2]];

  // u
  dofs[0] = 3 * node1 + 0;
  dofs[1] = 3 * node2 + 0;
  dofs[2] = 3 * node3 + 0;

  // v
  dofs[3] = 3 * node1 + 1;
  dofs[4] = 3 * node2 + 1;
  dofs[5] = 3 * node3 + 1;

  // w
  dofs[6] = 3 * node1 + 2;
  dofs[7] = 3 * node2 + 2;
  dofs[8] = 3 * node3 + 2;

  // rotation
  dofs[9]  = manifold.EdgeDofNumber()[face_index][0]; // 12
  dofs[10] = manifold.EdgeDofNumber()[face_index][1]; // 23
  dofs[11] = manifold.EdgeDofNumber()[face_index][2]; // 13

  return dofs;
}

void FiniteElementAnalysis::add_pressure_loads(Eigen::VectorXd& forces) const {

  for (const auto& c : pressure_manifolds_) {
    const auto m             = c.first;
    const Manifold& manifold = mesh_.manifolds[m];
    const auto pressure      = c.second;

    for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto force = ei.compute_pressure_force(coordinates, pressure);
      const auto dofs  = compute_local_dof(m, i);

      forces(dofs[0]) += force(0);
      forces(dofs[1]) += force(0);
      forces(dofs[2]) += force(0);

      forces(dofs[3]) += force(1);
      forces(dofs[4]) += force(1);
      forces(dofs[5]) += force(1);

      forces(dofs[6]) += force(2);
      forces(dofs[7]) += force(2);
      forces(dofs[8]) += force(2);
    }
  }
}

void FiniteElementAnalysis::add_pressure_shape_gradients(const Eigen::VectorXd& displacement,
                                                         std::vector<double>& gradient) const {

  for (const auto& c : pressure_manifolds_) {
    const auto m             = c.first;
    const Manifold& manifold = mesh_.manifolds[m];
    const auto pressure      = c.second;

    for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

      const auto& f = manifold.Faces()[i];

      std::array<Eigen::Vector3d, 3> coordinates;
      coordinates[0] = {manifold.Points()[f[0]][0], manifold.Points()[f[0]][1], manifold.Points()[f[0]][2]};
      coordinates[1] = {manifold.Points()[f[1]][0], manifold.Points()[f[1]][1], manifold.Points()[f[1]][2]};
      coordinates[2] = {manifold.Points()[f[2]][0], manifold.Points()[f[2]][1], manifold.Points()[f[2]][2]};

      const auto dofs = compute_local_dof(m, i);

      Eigen::Vector<double, 12> u_element;
      for (int j = 0; j < 12; j++)
        u_element(j) = displacement(dofs[j]);

      const auto grad = ei.compute_pressure_gradients(coordinates, pressure, u_element);

      const int ni1 = manifold.UniqueNodeNumber()[f[0]];
      const int ni2 = manifold.UniqueNodeNumber()[f[1]];
      const int ni3 = manifold.UniqueNodeNumber()[f[2]];

      gradient[3 * ni1 + 0] += grad[0];
      gradient[3 * ni1 + 1] += grad[1];
      gradient[3 * ni1 + 2] += grad[2];
      gradient[3 * ni2 + 0] += grad[3];
      gradient[3 * ni2 + 1] += grad[4];
      gradient[3 * ni2 + 2] += grad[5];
      gradient[3 * ni3 + 0] += grad[6];
      gradient[3 * ni3 + 1] += grad[7];
      gradient[3 * ni3 + 2] += grad[8];
    }
  }
}

void FiniteElementAnalysis::assemble_stiffness_matrix(Eigen::SparseMatrix<double>& mat) const {

  if (mat.nonZeros() == 0) {

    std::vector<Eigen::Triplet<double>> t;
    t.reserve(12 * 12 * mesh_.number_of_triangles());

    for (size_t m = 0; m < mesh_.manifolds.size(); m++) {

#pragma omp parallel for schedule(static) default(none) shared(t) firstprivate(m)
      for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

        const auto ke   = compute_local_stiffness(m, i);
        const auto dofs = compute_local_dof(m, i);

        // loop column first, as eigen matrices are stored column-major
#pragma omp critical
        {
          for (int jj = 0; jj < 12; jj++) {
            for (int ii = 0; ii < 12; ii++) {
              t.emplace_back(dofs[ii], dofs[jj], ke(ii, jj));
            }
          }
        }
      }
    }

    mat.setFromTriplets(t.begin(), t.end());
    mat.makeCompressed();
  } else {
    mat *= 0.0;

    for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
      for (size_t i = 0; i < mesh_.manifolds[m].Faces().size(); i++) {

        const auto ke   = compute_local_stiffness(m, i);
        const auto dofs = compute_local_dof(m, i);

        // loop column first, as eigen matrices are stored column-major
        for (int jj = 0; jj < 12; jj++) {
          for (int ii = 0; ii < 12; ii++) {
            mat.coeffRef(dofs[ii], dofs[jj]) += ke(ii, jj);
          }
        }
      }
    }
  }
}

void FiniteElementAnalysis::apply_boundary_conditions(Eigen::SparseMatrix<double>& mat, Eigen::VectorXd& force) const {

  for (const auto& fd : prescribedDofs) {
    force -= fd.second * mat.col(fd.first);
  }

  Eigen::VectorXd nullMatrix  = Eigen::VectorXd::Ones(mesh_.numberOfDofs);
  Eigen::VectorXd inverseNull = Eigen::VectorXd::Zero(mesh_.numberOfDofs);

  for (const auto& fd : fixedDofs) {
    nullMatrix.coeffRef(fd)  = 0.0;
    inverseNull.coeffRef(fd) = 1.0;
    force(fd)                = 0.0;
  }

  for (const auto& fd : prescribedDofs) {
    nullMatrix.coeffRef(fd.first)  = 0.0;
    inverseNull.coeffRef(fd.first) = 1.0;
    force(fd.first)                = fd.second;
  }

  mat = (nullMatrix.asDiagonal() * mat * nullMatrix.asDiagonal());
  mat += inverseNull.asDiagonal();
}

std::array<bool, 3> FiniteElementAnalysis::compute_rotation_flips(std::array<uint32_t, 3> ni) {
  return {ni[0] > ni[1], ni[1] > ni[2], ni[2] > ni[0]};
}