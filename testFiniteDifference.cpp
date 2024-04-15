#include "catch2/catch_test_macros.hpp"

#include "catch2/catch_approx.hpp"

#include "exodusWriter.h"
#include "finiteElementAnalysis.h"
#include "mesh.h"
#include "pdeFilter.h"
#include "qualityConstraint.h"
#include "thicknessFilter.h"
#include "volumeConstraint.h"

#include <iostream>

Mesh setup_plate_mesh() {
  Mesh plateMesh;
  plateMesh.read_ply_files("../testMeshes/unordered_plate*.ply", 0.1);
  plateMesh.setup_dof_numbering();
  return plateMesh;
}

const Mesh plateMesh = setup_plate_mesh();
const VolumeConstraint vc(plateMesh);

TEST_CASE("Volume computation") {
  const double volume = vc.compute_volume(plateMesh);
  REQUIRE(volume == Catch::Approx(0.1 * 1.0 * 1.0).margin(1e-10));
}

TEST_CASE("Volume constraint finite difference wrt shape") {
  std::cout << "Finite difference test of volume wrt shape" << std::endl;

  Mesh plateMesh = setup_plate_mesh();
  const VolumeConstraint vc(plateMesh);

  const std::vector<double> perturbations = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

  std::vector<double> perturbed_movement(plateMesh.numberOfUniqueNodes * 3, 0.0);
  double d = 0.0;
  for (auto& v : perturbed_movement) {
    v += d;
    d += 0.001;
  }

  plateMesh.move_nodes(perturbed_movement);
  const auto dg_nominal = vc.compute_shape_gradient();

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;
    for (size_t n = 0; n < 3 * plateMesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      plateMesh.move_nodes(perturbed_movement);
      const double g_perturbed_plus = vc.compute_value();
      perturbed_movement[n] -= 2.0 * perturbation;
      plateMesh.move_nodes(perturbed_movement);
      const double g_perturbed_minus = vc.compute_value();
      const double fd_gradient       = (g_perturbed_plus - g_perturbed_minus) / (2.0 * perturbation);
      REQUIRE(dg_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}

TEST_CASE("Volume constraint finite difference wrt thickness") {
  std::cout << "Finite difference test of volume wrt thickness" << std::endl;

  Mesh plateMesh = setup_plate_mesh();
  const VolumeConstraint vc(plateMesh);

  const std::vector<double> perturbations = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8};

  const double g_nominal = vc.compute_value();
  const auto dg_nominal  = vc.compute_thickness_gradient();

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;
    for (size_t m = 0; m < plateMesh.manifolds.size(); m++) {
      auto& manifold           = plateMesh.manifolds[m];
      auto perturbed_thickness = manifold.Thickness();

      for (size_t n = 0; n < manifold.Points().size(); n++) {
        perturbed_thickness[n] += perturbation;
        manifold.ThicknessMutable() = perturbed_thickness;
        const double g_perturbed    = vc.compute_value();
        const double fd_gradient    = (g_perturbed - g_nominal) / perturbation;
        REQUIRE(fd_gradient == Catch::Approx(dg_nominal[m][n]).margin(10 * perturbation));

        // reset thicknesses to original state
        perturbed_thickness[n] -= perturbation;
      }
      manifold.ThicknessMutable() = perturbed_thickness;
    }
  }
}

TEST_CASE("Volume constraint finite difference wrt thickness with filter") {
  std::cout << "Finite difference test of volume wrt thickness with filter" << std::endl;

  Mesh plateMesh = setup_plate_mesh();
  const VolumeConstraint vc(plateMesh);

  const std::vector<double> radii         = {0.0, 0.05, 0.1, 0.5, 1.0};
  const std::vector<double> perturbations = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7};

  for (auto radius : radii) {
    ThicknessFilter filter(plateMesh, radius);

    const double g_nominal = vc.compute_value();
    const auto dg_nominal  = filter.backward(vc.compute_thickness_gradient());

    std::vector<std::vector<double>> initialThickness(plateMesh.manifolds.size());
    std::vector<std::vector<double>> perturbedThk(plateMesh.manifolds.size());

    for (size_t m = 0; m < plateMesh.manifolds.size(); m++) {
      initialThickness[m] = plateMesh.manifolds[m].Thickness();
      perturbedThk[m]     = plateMesh.manifolds[m].Thickness();
    }

    double d = 0.0;
    for (size_t m = 0; m < plateMesh.manifolds.size(); m++) {
      for (auto& v : initialThickness[m]) {
        v += d;
        d += 0.001;
      }

      plateMesh.manifolds[m].ThicknessMutable() = initialThickness[m];
    }

    for (const auto perturbation : perturbations) {
      std::cout << "Testing perturbation " << perturbation << " radius " << radius << std::endl;
      for (size_t m = 0; m < plateMesh.manifolds.size(); m++) {
        auto& manifold = plateMesh.manifolds[m];

        for (size_t n = 0; n < manifold.Points().size(); n++) {
          perturbedThk[m][n] += perturbation;
          manifold.ThicknessMutable() = filter.forward(perturbedThk)[m];
          const double g_perturbed    = vc.compute_value();
          const double fd_gradient    = (g_perturbed - g_nominal) / perturbation;
          REQUIRE(fd_gradient == Catch::Approx(dg_nominal[m][n]).margin(100 * perturbation));

          // reset thicknesses to original state
          perturbedThk[m][n] -= perturbation;
        }
        manifold.ThicknessMutable() = initialThickness[m];
      }
    }
  }
}

TEST_CASE("Compliance wrt thickness finite difference") {
  std::cout << "Finite difference test of compliance wrt thickness" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/fd_mesh*.ply", 0.1);
  mesh.setup_dof_numbering();
  FiniteElementAnalysis fea(mesh);

  double force = -1.0;
  std::vector<std::pair<uint32_t, double>> forces;
  std::vector<uint32_t> bcDof;
  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1 = pt[0] < -0.499;
    const bool isOnB2 = pt[0] > 0.499;
    const bool isOnB3 = pt[1] < -0.499;
    const bool isOnB4 = pt[1] > 0.499;
    const bool isOnB5 = pt[2] < 0.001;

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4 || isOnB5) {
      bcDof.push_back(3 * i + 2);
    }

    if (isOnB5 && isOnB1 && isOnB3) {
      bcDof.push_back(3 * i + 0);
      bcDof.push_back(3 * i + 1);
    }

    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);
    if (isInCenter) {
      forces.push_back({3 * i + 2, force});
    }
  }

  fea.set_fixed_dof(bcDof);
  fea.set_forces(forces);

  const auto u_nominal   = fea.solve_system();
  const double c_nominal = fea.compute_compliance(u_nominal);
  const auto dc_nominal  = fea.compute_thickness_gradient(u_nominal);

  const std::vector<double> perturbations = {1e-5, 1e-7};

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;

    for (size_t m = 0; m < mesh.manifolds.size(); m++) {
      auto& manifold           = mesh.manifolds[m];
      auto perturbed_thickness = manifold.Thickness();

      for (size_t n = 0; n < manifold.Points().size(); n++) {
        perturbed_thickness[n] += perturbation;
        manifold.ThicknessMutable() = perturbed_thickness;
        const auto u                = fea.solve_system();
        const double c_perturbed    = fea.compute_compliance(u);
        const double fd_gradient    = (c_perturbed - c_nominal) / perturbation;
        REQUIRE(dc_nominal[m][n] == Catch::Approx(fd_gradient).margin(1e-2));

        // reset thicknesses to original state
        perturbed_thickness[n] -= perturbation;
      }
      manifold.ThicknessMutable() = perturbed_thickness;
    }
  }
}

TEST_CASE("Compliance wrt node position finite difference", "[pos]") {
  std::cout << "Finite difference test of compliance wrt position" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/fd_mesh*.ply", 0.1);
  mesh.setup_dof_numbering();
  FiniteElementAnalysis fea(mesh);

  double force = -1.0;
  std::vector<std::pair<uint32_t, double>> forces;
  std::vector<uint32_t> bcDof;
  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1 = pt[0] < -0.499;
    const bool isOnB2 = pt[0] > 0.499;
    const bool isOnB3 = pt[1] < -0.499;
    const bool isOnB4 = pt[1] > 0.499;
    const bool isOnB5 = pt[2] < 0.001;

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4 || isOnB5) {
      bcDof.push_back(3 * i + 2);
    }

    if (isOnB5 && isOnB1 && isOnB3) {
      bcDof.push_back(3 * i + 0);
      bcDof.push_back(3 * i + 1);
    }

    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);
    if (isInCenter) {
      forces.push_back({3 * i + 2, force});
    }
  }

  fea.set_fixed_dof(bcDof);
  fea.set_forces(forces);

  std::vector<double> node_movement(mesh.numberOfUniqueNodes * 3, 0.0);
  auto perturbed_movement = node_movement;

  const auto u_nominal  = fea.solve_system();
  const auto dc_nominal = fea.compute_shape_gradient(u_nominal);

  const std::vector<double> perturbations = {1e-5, 1e-7};

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;

    for (size_t n = 0; n < 3 * mesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      mesh.move_nodes(perturbed_movement);
      auto u                        = fea.solve_system();
      const double c_perturbed_plus = fea.compute_compliance(u);
      perturbed_movement[n] -= 2.0 * perturbation;
      mesh.move_nodes(perturbed_movement);
      u                              = fea.solve_system();
      const double c_perturbed_minus = fea.compute_compliance(u);
      const double fd_gradient       = (c_perturbed_plus - c_perturbed_minus) / (2.0 * perturbation);
      REQUIRE(dc_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}

TEST_CASE("Volume constraint finite difference wrt shape with filter", "[filter]") {
  std::cout << "Finite difference test of volume wrt shape with filter" << std::endl;

  Mesh plateMesh = setup_plate_mesh();
  const VolumeConstraint vc(plateMesh);
  std::shared_ptr<std::vector<uint32_t>> bound_ptr = std::make_shared<std::vector<uint32_t>>();
  PdeFilter filter(plateMesh, 0.05, {bound_ptr, bound_ptr, bound_ptr});

  const std::vector<double> perturbations = {1e-3, 1e-5, 1e-7};

  std::vector<double> perturbed_movement(plateMesh.numberOfUniqueNodes * 3, 0.0);

  plateMesh.move_nodes(filter.forward(perturbed_movement));
  const auto dg_nominal = filter.backward(vc.compute_shape_gradient());

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;
    for (size_t n = 0; n < 3 * plateMesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      plateMesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_plus = vc.compute_value();
      perturbed_movement[n] -= 2.0 * perturbation;
      plateMesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_minus = vc.compute_value();
      const double fd_gradient       = (g_perturbed_plus - g_perturbed_minus) / (2.0 * perturbation);
      REQUIRE(dg_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}

TEST_CASE("Volume constraint finite difference wrt shape with filter on multiple manifolds", "[filter]") {
  std::cout << "Finite difference test of volume wrt shape with filter on multiple manifolds" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/fd_multiple_manifolds*.ply", 0.01);
  mesh.setup_dof_numbering();

  std::vector<double> initial_position(mesh.numberOfUniqueNodes * 3, 0.0);
  for (const auto& m : mesh.manifolds)
    for (size_t i = 0; i < m.Points().size(); i++) {
      const auto ni                = m.UniqueNodeNumber()[i];
      initial_position[3 * ni + 0] = m.Points()[i][0];
      initial_position[3 * ni + 1] = m.Points()[i][1];
      initial_position[3 * ni + 2] = m.Points()[i][2];
    }

  const VolumeConstraint vc(mesh);
  std::shared_ptr<std::vector<uint32_t>> bound_ptr = std::make_shared<std::vector<uint32_t>>();
  PdeFilter filter(mesh, 0.1, {bound_ptr, bound_ptr, bound_ptr});

  const std::vector<double> perturbations = {1e-3, 1e-5, 1e-7};

  std::vector<double> perturbed_movement(mesh.numberOfUniqueNodes * 3, 0.0);
  double d = 0.0;
  for (auto& v : perturbed_movement) {
    v += d;
    d += 0.001;
  }

  const auto fpos = filter.forward(perturbed_movement);
  mesh.move_nodes(fpos);

  const auto dg_nominal = filter.backward(vc.compute_shape_gradient());

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;
    for (size_t n = 0; n < 3 * mesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      mesh.set_nodes(initial_position);
      mesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_plus = vc.compute_value();
      perturbed_movement[n] -= 2.0 * perturbation;
      mesh.set_nodes(initial_position);
      mesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_minus = vc.compute_value();
      const double fd_gradient       = (g_perturbed_plus - g_perturbed_minus) / (2.0 * perturbation);

      REQUIRE(dg_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}

TEST_CASE("Aspect ratio constraint finite difference wrt shape with filter on multiple manifolds", "[filter]") {
  std::cout << "Finite difference test of AR wrt shape with filter on multiple manifolds" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/fd_multiple_manifolds*.ply", 0.01);
  mesh.setup_dof_numbering();

  const QualityConstraint qc(mesh);
  std::shared_ptr<std::vector<uint32_t>> bound_ptr = std::make_shared<std::vector<uint32_t>>();
  PdeFilter filter(mesh, 0.5, {bound_ptr, bound_ptr, bound_ptr});

  const std::vector<double> perturbations = {1e-3, 1e-5, 1e-7};

  const auto dg         = qc.compute_AR_shape_gradient();
  const auto dg_nominal = filter.backward(dg);

  std::vector<double> node_movement(mesh.numberOfUniqueNodes * 3, 0.0);
  auto perturbed_movement = node_movement;

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;
    for (size_t n = 0; n < 3 * mesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      mesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_plus = qc.compute_AR_value();
      perturbed_movement[n] -= 2.0 * perturbation;
      mesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_minus = qc.compute_AR_value();
      const double fd_gradient       = (g_perturbed_plus - g_perturbed_minus) / (2.0 * perturbation);

      REQUIRE(dg_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}

TEST_CASE("Volume constraint finite difference wrt shape with PDE filter on multiple manifolds", "[filter]") {
  std::cout << "Finite difference test of volume wrt shape with PDE filter on multiple manifolds" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/fd_multiple_manifolds*.ply", 0.01);
  mesh.setup_dof_numbering();

  std::vector<double> initial_position(mesh.numberOfUniqueNodes * 3, 0.0);
  std::vector<uint32_t> bound_nodes;
  for (const auto& m : mesh.manifolds)
    for (size_t i = 0; i < m.Points().size(); i++) {
      const auto ni                = m.UniqueNodeNumber()[i];
      initial_position[3 * ni + 0] = m.Points()[i][0];
      initial_position[3 * ni + 1] = m.Points()[i][1];
      initial_position[3 * ni + 2] = m.Points()[i][2];

      if (m.Points()[i][0] < 0.001) {
        bound_nodes.push_back(m.UniqueNodeNumber()[i]);
      }
    }

  const VolumeConstraint vc(mesh);
  std::shared_ptr<std::vector<uint32_t>> bound_ptr = std::make_shared<std::vector<uint32_t>>(bound_nodes);
  PdeFilter filter(mesh, 0.5, {bound_ptr, bound_ptr, bound_ptr});

  const std::vector<double> perturbations = {1e-3, 1e-5, 1e-7};

  std::vector<double> perturbed_movement(mesh.numberOfUniqueNodes * 3, 0.0);
  double d = 0.0;
  for (auto& v : perturbed_movement) {
    v += d;
    d += 0.001;
  }

  const auto fpos = filter.forward(perturbed_movement);
  mesh.move_nodes(fpos);

  const auto dg_nominal = filter.backward(vc.compute_shape_gradient());

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;
    for (size_t n = 0; n < 3 * mesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      mesh.set_nodes(initial_position);
      mesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_plus = vc.compute_value();
      perturbed_movement[n] -= 2.0 * perturbation;
      mesh.set_nodes(initial_position);
      mesh.move_nodes(filter.forward(perturbed_movement));
      const double g_perturbed_minus = vc.compute_value();
      const double fd_gradient       = (g_perturbed_plus - g_perturbed_minus) / (2.0 * perturbation);

      REQUIRE(dg_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}

TEST_CASE("Compliance wrt node position with pressure load finite difference") {
  std::cout << "Pressure load" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/fd_mesh*.ply", 0.1);
  mesh.setup_dof_numbering();
  FiniteElementAnalysis fea(mesh);

  double force = -1.0;
  std::vector<std::pair<uint32_t, double>> forces;
  std::vector<uint32_t> bcDof;
  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1 = pt[0] < -0.499;
    const bool isOnB2 = pt[0] > 0.499;
    const bool isOnB3 = pt[1] < -0.499;
    const bool isOnB4 = pt[1] > 0.499;
    const bool isOnB5 = pt[2] < 0.001;

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4 || isOnB5) {
      bcDof.push_back(3 * i + 2);
    }

    if (isOnB5 && isOnB1 && isOnB3) {
      bcDof.push_back(3 * i + 0);
      bcDof.push_back(3 * i + 1);
    }

    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);
    if (isInCenter) {
      forces.push_back({3 * i + 2, force});
    }
  }

  fea.set_fixed_dof(bcDof);
  fea.set_forces(forces);

  std::vector<double> node_movement(mesh.numberOfUniqueNodes * 3, 0.0);
  auto perturbed_movement = node_movement;

  const auto u_nominal  = fea.solve_system();
  const auto dc_nominal = fea.compute_shape_gradient(u_nominal);

  const std::vector<double> perturbations = {1e-7};

  for (const auto perturbation : perturbations) {
    std::cout << "Testing perturbation " << perturbation << std::endl;

    for (size_t n = 0; n < 3 * mesh.numberOfUniqueNodes; n++) {

      perturbed_movement[n] += perturbation;
      mesh.move_nodes(perturbed_movement);
      auto u                        = fea.solve_system();
      const double c_perturbed_plus = fea.compute_compliance(u);
      perturbed_movement[n] -= 2.0 * perturbation;
      mesh.move_nodes(perturbed_movement);
      u                              = fea.solve_system();
      const double c_perturbed_minus = fea.compute_compliance(u);
      const double fd_gradient       = (c_perturbed_plus - c_perturbed_minus) / (2.0 * perturbation);
      REQUIRE(dc_nominal[n] == Catch::Approx(fd_gradient).margin(1e-2));

      // reset value to original state
      perturbed_movement[n] += perturbation;
    }
  }
}