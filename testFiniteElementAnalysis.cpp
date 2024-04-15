#include "catch2/catch_test_macros.hpp"

#include "catch2/catch_approx.hpp"

#include <iostream>

#include "exodusWriter.h"
#include "finiteElementAnalysis.h"
#include "manifold.h"
#include "mesh.h"
#include "shellElement.h"

// setup methods
Manifold make_square_plate(const int plane, const std::array<double, 2> domainSize,
                           const std::array<int, 2> domainElements, const double thickness) {

  std::vector<std::array<double, 3>> points;
  std::vector<std::array<uint32_t, 3>> faces;

  if (plane == 0) {
    double length = domainSize[0];
    double width  = domainSize[1];

    size_t nLength = domainElements[0];
    size_t nWidth  = domainElements[1];

    double yDist = length / nLength;
    double xDist = width / nWidth;

    // setup points at bottom
    points.reserve((nWidth + 1) * (nLength + 1));
    for (size_t i = 0; i < nWidth + 1; i++)
      for (size_t j = 0; j < nLength + 1; j++) {
        double xpos = i * xDist;
        double ypos = j * yDist;
        points.push_back({xpos, ypos, 0});
      }

    // setup faces at bottom
    faces.reserve(2 * nWidth * nLength);

    for (size_t i = 0; i < nWidth; i++)
      for (size_t j = 0; j < nLength; j++) {
        uint32_t idx1 = i * (nLength + 1) + j;
        uint32_t idx2 = i * (nLength + 1) + j + 1;
        uint32_t idx3 = (i + 1) * (nLength + 1) + j + 1;
        uint32_t idx4 = (i + 1) * (nLength + 1) + j;

        faces.push_back({idx1, idx2, idx4});
        faces.push_back({idx2, idx3, idx4});
      }
  }

  return Manifold(points, faces, thickness);
}

Mesh make_square_plate_mesh(const int plane, const std::array<double, 2> domainSize,
                            const std::array<int, 2> domainElements, const double thickness) {
  Mesh mesh;
  mesh.manifolds.push_back(make_square_plate(plane, domainSize, domainElements, thickness));

  mesh.setup_dof_numbering();

  return mesh;
}

TEST_CASE("Square plate bending with point load", "[plate_bending_point]") {

  std::cout << std::endl << "Plate bending point load" << std::endl;

  const int nelx         = 20;
  const int nely         = 20;
  const double thickness = 0.1;
  Mesh plateMesh         = make_square_plate_mesh(0, {1, 1}, {nelx, nely}, thickness);
  FiniteElementAnalysis fea(plateMesh);

  std::vector<uint32_t> bcDof;
  for (size_t i = 0; i < nelx + 1; i++) {
    int j1 = 0;
    int j2 = nely;
    int n1 = i * (nely + 1) + j1;
    int n2 = i * (nely + 1) + j2;
    bcDof.push_back(3 * n1 + 2);
    bcDof.push_back(3 * n2 + 2);
    bcDof.push_back(3 * n2 + 0);
    bcDof.push_back(3 * n2 + 0);
    bcDof.push_back(3 * n2 + 1);
    bcDof.push_back(3 * n2 + 1);
  }
  for (size_t j = 1; j < nely; j++) {
    int i1 = 0;
    int i2 = nelx;
    int n1 = i1 * (nely + 1) + j;
    int n2 = i2 * (nely + 1) + j;
    bcDof.push_back(3 * n1 + 2);
    bcDof.push_back(3 * n2 + 2);
  }
  fea.set_fixed_dof(bcDof);

  int centerx   = (nelx / 2);
  int centery   = (nely / 2);
  int centern   = centerx * (nely + 1) + centery;
  int centerDof = 3 * centern + 2;

  double force = -1.0;
  std::vector<std::pair<uint32_t, double>> forces;
  forces.push_back({centerDof, force});
  fea.set_forces(forces);

  auto displacement = fea.solve_system();

  const double solutionAtCenter = displacement(centerDof);
  std::cout << "FEA solution " << solutionAtCenter << std::endl;

  // Roarks formulas for stress and strain, 7.ed., p502
  const double analyticSolution = 0.1267 * force / (thickness * thickness * thickness);
  std::cout << "Analytic solution " << analyticSolution << std::endl;

  // check that solution is within 5% of analytic solution
  REQUIRE(solutionAtCenter == Catch::Approx(analyticSolution).margin(std::abs(0.05 * analyticSolution)));
}

TEST_CASE("Square plate bending with distributed load", "[plate_bending_dist]") {

  std::cout << std::endl << "Plate bending distributed load" << std::endl;

  const int nelx         = 20;
  const int nely         = 20;
  const double thickness = 0.1;
  Mesh plateMesh         = make_square_plate_mesh(0, {1, 1}, {nelx, nely}, thickness);
  FiniteElementAnalysis fea(plateMesh);

  std::vector<uint32_t> bcDof;
  for (size_t i = 0; i < nelx + 1; i++) {
    int j1 = 0;
    int j2 = nely;
    int n1 = i * (nely + 1) + j1;
    int n2 = i * (nely + 1) + j2;
    bcDof.push_back(3 * n1 + 2);
    bcDof.push_back(3 * n2 + 2);
    bcDof.push_back(3 * n1 + 0);
    bcDof.push_back(3 * n1 + 1);
  }
  for (size_t j = 1; j < nely; j++) {
    int i1 = 0;
    int i2 = nelx;
    int n1 = i1 * (nely + 1) + j;
    int n2 = i2 * (nely + 1) + j;
    bcDof.push_back(3 * n1 + 2);
    bcDof.push_back(3 * n2 + 2);
  }
  fea.set_fixed_dof(bcDof);

  int centerx   = (nelx / 2);
  int centery   = (nely / 2);
  int centern   = centerx * (nely + 1) + centery;
  int centerDof = 3 * centern + 2;

  double point_force = -1.0 / (nelx * nely);
  std::vector<std::pair<uint32_t, double>> forces;
  for (size_t i = 1; i < nelx; i++)
    for (size_t j = 1; j < nely; j++) {
      int n   = i * (nely + 1) + j;
      int dof = 3 * n + 2;
      forces.push_back({dof, point_force});
    }
  fea.set_forces(forces);

  auto displacement = fea.solve_system();

  const double solutionAtCenter = displacement(centerDof);
  std::cout << "FEA solution " << solutionAtCenter << std::endl;

  // Roarks formulas for stress and strain, 7.ed., p502
  const double analyticSolution = 0.0444 * -1.0 / (thickness * thickness * thickness);
  std::cout << "Analytic solution " << analyticSolution << std::endl;

  // check that solution is within 1% of analytic solution
  REQUIRE(solutionAtCenter == Catch::Approx(analyticSolution).margin(std::abs(0.01 * analyticSolution)));
}

TEST_CASE("Square membrane in normal tension") {

  std::cout << std::endl << "Plate bending distributed load" << std::endl;

  const int nelx         = 20;
  const int nely         = 20;
  const double thickness = 0.1;
  Mesh plateMesh         = make_square_plate_mesh(0, {1, 1}, {nelx, nely}, thickness);
  FiniteElementAnalysis fea(plateMesh);

  std::vector<uint32_t> bcDof;
  for (size_t i = 0; i < nelx + 1; i++) {
    int j = 0;
    int n = i * (nely + 1) + j;
    bcDof.push_back(3 * n + 1);
    bcDof.push_back(3 * n + 2);
  }
  int centerx = (nelx / 2);
  int ncenter = centerx * (nely + 1) + 0;
  bcDof.push_back(3 * ncenter + 0);
  fea.set_fixed_dof(bcDof);

  double point_force = 1.0 / (nelx);
  double totalForce  = 0.0;
  std::vector<std::pair<uint32_t, double>> forces;
  for (size_t i = 0; i < nelx + 1; i++) {
    int j            = nely;
    int n            = i * (nely + 1) + j;
    int dof          = 3 * n + 1;
    double thisForce = point_force;
    if (i == 0 || i == nelx)
      thisForce = point_force * 0.5;
    forces.push_back({dof, thisForce});
    totalForce += thisForce;
  }
  fea.set_forces(forces);

  auto displacement = fea.solve_system();

  const double analyticDisplacement = (totalForce * 1.0) / (thickness * 1.0 * 1.0); // F*L / E*area
  std::cout << "Analytic solution " << analyticDisplacement << std::endl;

  for (size_t i = 0; i < nelx + 1; i++) {
    int j   = nely;
    int n   = i * (nely + 1) + j;
    int dof = 3 * n + 1;

    double displacementAtBorder = displacement(dof);

    // check that solution is within 1% of analytic solution
    REQUIRE(displacementAtBorder == Catch::Approx(analyticDisplacement).margin(1e-3));
  }

  // check that the contraction matches the poissons ratio
  const double analyticContraction = 0.3 * 0.5 * analyticDisplacement; // scale with poissons ratio, and half length.
  int n1                           = 0 * (nely + 1) + nely;
  int n2                           = nelx * (nely + 1) + nely;
  int dof1                         = 3 * n1 + 0;
  int dof2                         = 3 * n2 + 0;
  REQUIRE(displacement(dof1) == Catch::Approx(analyticContraction).margin(1e-3));
  REQUIRE(displacement(dof2) == Catch::Approx(-analyticContraction).margin(1e-3));
}

TEST_CASE("Unordered square plate bending with point load", "[plate_bending_point]") {

  std::cout << std::endl << "Plate bending point load with unordered mesh" << std::endl;

  Mesh mesh;
  const double thickness = 0.1;

  mesh.read_ply_files("../testMeshes/unordered_plate*.ply", thickness);
  mesh.setup_dof_numbering();
  FiniteElementAnalysis fea(mesh);

  const auto& manifold = mesh.manifolds[0];

  double force = -1.0;
  std::vector<std::pair<uint32_t, double>> forces;
  std::vector<uint32_t> bcDof;
  int centerDof = 0;
  for (size_t i = 0; i < manifold.Points().size(); i++) {
    const auto& pt = manifold.Points()[i];

    const bool isOnB1 = pt[0] < -0.499;
    const bool isOnB2 = pt[0] > 0.499;
    const bool isOnB3 = pt[1] < -0.499;
    const bool isOnB4 = pt[1] > 0.499;

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4) {
      bcDof.push_back(3 * i + 2);
      bcDof.push_back(3 * i + 0);
      bcDof.push_back(3 * i + 1);
    }

    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);
    if (isInCenter) {
      centerDof = 3 * i + 2;
      forces.push_back({centerDof, force});
    }
  }

  fea.set_fixed_dof(bcDof);
  fea.set_forces(forces);

  auto displacement = fea.solve_system();

  const double solutionAtCenter = displacement(centerDof);
  std::cout << "FEA solution " << solutionAtCenter << std::endl;

  // Roarks formulas for stress and strain, 7.ed., p502
  const double analyticSolution = 0.1267 * force / (thickness * thickness * thickness);
  std::cout << "Analytic solution " << analyticSolution << std::endl;

  // check that solution is within 5% of analytic solution
  REQUIRE(solutionAtCenter == Catch::Approx(analyticSolution).margin(std::abs(0.05 * analyticSolution)));
}

// test bending or membrane in non-axis aligned plate
TEST_CASE("Unordered non-aligned square plate bending with point load", "[plate_bending_point]") {

  std::cout << std::endl << "Plate bending point load with unordered mesh and non-trivial orientation" << std::endl;

  Mesh mesh;
  const double thickness = 0.1;

  mesh.read_ply_files("../testMeshes/unordered_plate*.ply", thickness);
  mesh.setup_dof_numbering();
  FiniteElementAnalysis fea(mesh);

  const auto& manifold = mesh.manifolds[0];

  std::vector<std::pair<uint32_t, double>> forces;
  std::vector<uint32_t> bcDof;
  int centerDofx = 0, centerDofy = 0, centerDofz = 0;
  for (size_t i = 0; i < manifold.Points().size(); i++) {
    const auto& pt = manifold.Points()[i];

    const bool isOnB1 = pt[0] < -0.499;
    const bool isOnB2 = pt[0] > 0.499;
    const bool isOnB3 = pt[1] < -0.499;
    const bool isOnB4 = pt[1] > 0.499;

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4) {
      bcDof.push_back(3 * i + 0);
      bcDof.push_back(3 * i + 1);
      bcDof.push_back(3 * i + 2);
    }

    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);
    if (isInCenter) {
      centerDofx = 3 * i + 0;
      centerDofy = 3 * i + 1;
      centerDofz = 3 * i + 2;
    }
  }
  fea.set_fixed_dof(bcDof);

  std::array<Eigen::Vector3d, 3> newOrientation;
  newOrientation[0] = {1, 1, 1};
  newOrientation[1] = {1, -0.5, 2};
  newOrientation[1] =
      newOrientation[1] -
      (newOrientation[0].dot(newOrientation[1]) / newOrientation[0].dot(newOrientation[0])) * newOrientation[0];
  newOrientation[2] = newOrientation[0].cross(newOrientation[1]);

  newOrientation[0].normalize();
  newOrientation[1].normalize();
  newOrientation[2].normalize();
  const auto T = Shell::compute_rotation_matrix(newOrientation);

  // rotate mesh
  std::vector<double> node_movements(3 * mesh.numberOfUniqueNodes);
  for (size_t i = 0; i < manifold.Points().size(); i++) {
    const auto& pt             = manifold.Points()[i];
    Eigen::Vector3d pv         = {pt[0], pt[1], pt[2]};
    pv                         = T * pv;
    size_t ni                  = manifold.UniqueNodeNumber()[i];
    node_movements[3 * ni + 0] = pv[0] - pt[0];
    node_movements[3 * ni + 1] = pv[1] - pt[1];
    node_movements[3 * ni + 2] = pv[2] - pt[2];
  }
  mesh.move_nodes(node_movements);

  double force                = -1.0;
  Eigen::Vector3d forceVector = {0.0, 0.0, -1.0};
  forceVector                 = T * forceVector;
  forces.push_back({centerDofx, forceVector(0)});
  forces.push_back({centerDofy, forceVector(1)});
  forces.push_back({centerDofz, forceVector(2)});
  fea.set_forces(forces);

  auto displacement = fea.solve_system();

  Eigen::Vector3d centerDisp = {displacement(centerDofx), displacement(centerDofy), displacement(centerDofz)};
  centerDisp                 = T.transpose() * centerDisp;

  const double solutionAtCenter = centerDisp(2);
  std::cout << "FEA solution " << solutionAtCenter << std::endl;

  // Roarks formulas for stress and strain, 7.ed., p502
  const double analyticSolution = 0.1267 * force / (thickness * thickness * thickness);
  std::cout << "Analytic solution " << analyticSolution << std::endl;

  // check that solution is within 5% of analytic solution
  REQUIRE(solutionAtCenter == Catch::Approx(analyticSolution).margin(std::abs(0.05 * analyticSolution)));
  REQUIRE(centerDisp(0) == Catch::Approx(0.0).margin(std::abs(1e-9)));
  REQUIRE(centerDisp(1) == Catch::Approx(0.0).margin(std::abs(1e-9)));
}

TEST_CASE("Pinched hemisphere", "[sphere]") {

  std::cout << std::endl << "Pinched hemisphere" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/hemisphere*.ply", 0.04);
  mesh.setup_dof_numbering();
  FiniteElementAnalysis fea(mesh);

  double force = 1.0;
  std::vector<std::pair<uint32_t, double>> forces;
  std::vector<uint32_t> bcDof;
  int loadDof = 0;
  for (const auto& m : mesh.manifolds)
    for (size_t i = 0; i < m.Points().size(); i++) {
      const auto& pt = m.Points()[i];

      // set displacement bc
      const bool isOnB1 = pt[0] < 0.001;
      const bool isOnB2 = pt[2] < 0.001;

      if (isOnB1) {
        bcDof.push_back(3 * i + 0);
      }

      if (isOnB2) {
        bcDof.push_back(3 * i + 2);
      }

      const bool isPointLoad1 = isOnB1 && (pt[1] < 0.001);
      const bool isPointLoad2 = isOnB2 && (pt[1] < 0.001);

      if (isPointLoad1) {
        loadDof = 3 * i + 2;
        forces.push_back({loadDof, force});
        bcDof.push_back(3 * i + 1);
      }
      if (isPointLoad2) {
        forces.push_back({3 * i + 0, -force});
      }
    }

  for (const auto& m : mesh.manifolds)
    for (size_t i = 0; i < m.Faces().size(); i++) {
      const auto& f         = m.Faces()[i];
      const bool pt1_isOnB1 = m.Points()[f[0]][0] < 0.001;
      const bool pt2_isOnB1 = m.Points()[f[1]][0] < 0.001;
      const bool pt3_isOnB1 = m.Points()[f[2]][0] < 0.001;

      const bool pt1_isOnB2 = m.Points()[f[0]][2] < 0.001;
      const bool pt2_isOnB2 = m.Points()[f[1]][2] < 0.001;
      const bool pt3_isOnB2 = m.Points()[f[2]][2] < 0.001;

      const bool pt1_isOnB = pt1_isOnB1 || pt1_isOnB2;
      const bool pt2_isOnB = pt2_isOnB1 || pt2_isOnB2;
      const bool pt3_isOnB = pt3_isOnB1 || pt3_isOnB2;

      if (pt1_isOnB && pt2_isOnB)
        bcDof.push_back(m.EdgeDofNumber()[i][0]);

      if (pt2_isOnB && pt3_isOnB)
        bcDof.push_back(m.EdgeDofNumber()[i][1]);

      if (pt1_isOnB && pt3_isOnB)
        bcDof.push_back(m.EdgeDofNumber()[i][2]);
    }

  fea.set_fixed_dof(bcDof);
  fea.set_forces(forces);

  auto displacement = fea.solve_system();

  // rescale to 'correct' solution due to linearity.
  const double E                = 6.825e7;
  const double solutionAtCenter = displacement(loadDof) / E;
  std::cout << "FEA solution " << solutionAtCenter << std::endl;

  // I-beam cantilever deflection = F * L^3 / 3EI @ x=L
  // I = 0.0000044933 m^4 for 0.1x0.1 cross-section with 0.01 thickness
  const double analyticSolution = 0.0924;
  std::cout << "Analytic solution " << analyticSolution << std::endl;

  std::cout << "Deviation " << (std::abs(solutionAtCenter - analyticSolution) / analyticSolution) * 100.0 << "%"
            << std::endl;

  // check that solution is within 1% of analytic solution
  REQUIRE(solutionAtCenter == Catch::Approx(analyticSolution).margin(std::abs(0.01 * analyticSolution)));
}

TEST_CASE("Rigid body motion", "[rbm]") {

  std::cout << std::endl << "Rigid body motion" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/unordered_plate*.ply", 0.1);
  mesh.setup_dof_numbering();

  const auto& manifold = mesh.manifolds[0];

  const double displacement = 0.0432423;
  std::vector<std::pair<uint32_t, double>> motion_x, motion_y, motion_z;
  std::vector<uint32_t> bc_x, bc_y, bc_z, bc_xyz, bc_rotation;
  for (size_t i = 0; i < manifold.Points().size(); i++) {
    const auto& pt = manifold.Points()[i];

    const bool isOnB1 = pt[0] < -0.499;

    if (isOnB1) {
      motion_x.emplace_back(3 * i + 0, displacement);
      motion_y.emplace_back(3 * i + 1, displacement);
      motion_z.emplace_back(3 * i + 2, displacement);
    }

    if (isOnB1) {
      bc_x.push_back(3 * i + 1);
      bc_x.push_back(3 * i + 2);

      bc_y.push_back(3 * i + 0);
      bc_y.push_back(3 * i + 2);

      bc_z.push_back(3 * i + 0);
      bc_z.push_back(3 * i + 1);

      bc_rotation.push_back(3 * i + 0);
      bc_rotation.push_back(3 * i + 1);
      bc_rotation.push_back(3 * i + 2);
    }
  }

  const double slope = 0.5;
  std::vector<std::pair<uint32_t, double>> motion_rotation;
  for (const auto& m : mesh.manifolds)
    for (size_t i = 0; i < m.Faces().size(); i++) {
      const auto& f        = m.Faces()[i];
      const bool pt1_isOnB = m.Points()[f[0]][0] < -0.499;
      const bool pt2_isOnB = m.Points()[f[1]][0] < -0.499;
      const bool pt3_isOnB = m.Points()[f[2]][0] < -0.499;

      if (pt1_isOnB && pt2_isOnB) {
        const auto dof = m.EdgeDofNumber()[i][0];
        bc_z.push_back(dof);
        bc_xyz.push_back(dof);
        bc_x.push_back(dof);
        bc_y.push_back(dof);
        int n1 = m.UniqueNodeNumber()[f[0]];
        int n2 = m.UniqueNodeNumber()[f[1]];
        if (n1 > n2)
          motion_rotation.emplace_back(dof, slope);
        else
          motion_rotation.emplace_back(dof, -slope);
      }

      if (pt2_isOnB && pt3_isOnB) {
        const auto dof = m.EdgeDofNumber()[i][1];
        bc_z.push_back(dof);
        bc_xyz.push_back(dof);
        bc_x.push_back(dof);
        bc_y.push_back(dof);
        int n2 = m.UniqueNodeNumber()[f[1]];
        int n3 = m.UniqueNodeNumber()[f[2]];
        if (n2 > n3)
          motion_rotation.emplace_back(dof, slope);
        else
          motion_rotation.emplace_back(dof, -slope);
      }

      if (pt1_isOnB && pt3_isOnB) {
        const auto dof = m.EdgeDofNumber()[i][2];
        bc_z.push_back(dof);
        bc_xyz.push_back(dof);
        int n1 = m.UniqueNodeNumber()[f[0]];
        int n3 = m.UniqueNodeNumber()[f[2]];
        if (n3 > n1)
          motion_rotation.emplace_back(dof, slope);
        else
          motion_rotation.emplace_back(dof, -slope);
      }
    }

  std::vector<std::pair<uint32_t, double>> motion_xyz;
  motion_xyz.insert(motion_xyz.end(), motion_x.begin(), motion_x.end());
  motion_xyz.insert(motion_xyz.end(), motion_y.begin(), motion_y.end());
  motion_xyz.insert(motion_xyz.end(), motion_z.begin(), motion_z.end());

  {
    FiniteElementAnalysis fea(mesh);
    fea.set_fixed_dof(bc_x);
    fea.set_prescribed_displacements(motion_x);
    auto solution = fea.solve_system();
    for (uint32_t i = 0; i < mesh.numberOfUniqueNodes; i++) {
      REQUIRE(solution(3 * i + 0) == Catch::Approx(displacement).margin(1e-14));
      REQUIRE(solution(3 * i + 1) == Catch::Approx(0.0).margin(1e-14));
      REQUIRE(solution(3 * i + 2) == Catch::Approx(0.0).margin(1e-14));
    }

    // test that all bending and membrane stresses are 0
    auto stress = fea.compute_stress(solution);
    for (const auto& s : stress)
      for (const auto& ss : s)
        for (const auto& sss : ss)
          for (const auto& ssss : sss) {
            REQUIRE(ssss == Catch::Approx(0.0).margin(1e-14));
          }
  }

  {
    FiniteElementAnalysis fea(mesh);
    fea.set_fixed_dof(bc_y);
    fea.set_prescribed_displacements(motion_y);
    auto solution = fea.solve_system();

    for (uint32_t i = 0; i < mesh.numberOfUniqueNodes; i++) {
      REQUIRE(solution(3 * i + 0) == Catch::Approx(0.0).margin(1e-14));
      REQUIRE(solution(3 * i + 1) == Catch::Approx(displacement).margin(1e-14));
      REQUIRE(solution(+2) == Catch::Approx(0.0).margin(1e-14));
    }

    // test that all bending and membrane stresses are 0
    const auto stress = fea.compute_stress(solution);
    for (const auto& s : stress)
      for (const auto& ss : s)
        for (const auto& sss : ss)
          for (const auto& ssss : sss) {
            REQUIRE(ssss == Catch::Approx(0.0).margin(1e-14));
          }
  }

  {
    FiniteElementAnalysis fea(mesh);
    fea.set_fixed_dof(bc_z);
    fea.set_prescribed_displacements(motion_z);
    const auto solution = fea.solve_system();

    for (uint32_t i = 0; i < mesh.numberOfUniqueNodes; i++) {
      REQUIRE(solution(3 * i + 0) == Catch::Approx(0.0).margin(1e-14));
      REQUIRE(solution(3 * i + 1) == Catch::Approx(0.0).margin(1e-14));
      REQUIRE(solution(3 * i + 2) == Catch::Approx(displacement).margin(1e-14));
    }

    // test that all bending and membrane stresses are 0
    const auto stress = fea.compute_stress(solution);
    for (const auto& s : stress)
      for (const auto& ss : s)
        for (const auto& sss : ss)
          for (const auto& ssss : sss) {
            REQUIRE(ssss == Catch::Approx(0.0).margin(1e-14));
          }
  }

  {
    FiniteElementAnalysis fea(mesh);
    fea.set_fixed_dof(bc_xyz);
    fea.set_prescribed_displacements(motion_xyz);
    const auto solution = fea.solve_system();
    for (uint32_t i = 0; i < mesh.numberOfUniqueNodes; i++) {
      REQUIRE(solution(3 * i + 0) == Catch::Approx(displacement).margin(1e-14));
      REQUIRE(solution(3 * i + 1) == Catch::Approx(displacement).margin(1e-14));
      REQUIRE(solution(3 * i + 2) == Catch::Approx(displacement).margin(1e-14));
    }

    // test that all bending and membrane stresses are 0
    const auto stress = fea.compute_stress(solution);
    for (const auto& s : stress)
      for (const auto& ss : s)
        for (const auto& sss : ss)
          for (const auto& ssss : sss) {
            REQUIRE(ssss == Catch::Approx(0.0).margin(1e-14));
          }
  }

  {
    FiniteElementAnalysis fea(mesh);
    fea.set_fixed_dof(bc_rotation);
    fea.set_prescribed_displacements(motion_rotation);
    const auto solution = fea.solve_system();
    for (uint32_t i = 0; i < mesh.numberOfUniqueNodes; i++) {
      const double dispAtPoint = (manifold.Points()[i][0] + 0.5) * slope;
      REQUIRE(solution(3 * i + 0) == Catch::Approx(0.0).margin(1e-14));
      REQUIRE(solution(3 * i + 1) == Catch::Approx(0.0).margin(1e-14));
      REQUIRE(solution(3 * i + 2) == Catch::Approx(dispAtPoint).margin(1e-14));
    }

    // test that all bending and membrane stresses are 0
    const auto stress = fea.compute_stress(solution);
    for (const auto& s : stress)
      for (const auto& ss : s)
        for (const auto& sss : ss)
          for (const auto& ssss : sss) {
            REQUIRE(ssss == Catch::Approx(0.0).margin(1e-14));
          }
  }
}

TEST_CASE("Constant bending", "[bend]") {

  std::cout << std::endl << "Constant bending" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/unordered_plate*.ply", 0.1);
  mesh.setup_dof_numbering();

  const auto& manifold = mesh.manifolds[0];

  std::vector<std::pair<uint32_t, double>> force;
  std::vector<uint32_t> bc;
  for (size_t i = 0; i < manifold.Points().size(); i++) {
    const auto& pt = manifold.Points()[i];

    const bool isOnB1     = pt[0] < -0.499;
    const bool isOnB2     = pt[1] < -0.499;
    const bool isOnB3     = pt[0] > 0.499;
    const bool isOnB4     = pt[1] > 0.499;
    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);

    if (isInCenter) {
      bc.push_back(3 * i + 0);
      bc.push_back(3 * i + 1);
      bc.push_back(3 * i + 2);
    }

    if (isOnB1 && (pt[1] < 0.001)) {
      bc.push_back(3 * i + 0);
      bc.push_back(3 * i + 1);
      bc.push_back(3 * i + 2);
    }

    if (isOnB1 && isOnB2) {
      force.emplace_back(3 * i + 2, 1.0);
    }

    if (isOnB2 && isOnB3) {
      force.emplace_back(3 * i + 2, -1.0);
    }

    if (isOnB3 && isOnB4) {
      force.emplace_back(3 * i + 2, 1.0);
    }

    if (isOnB4 && isOnB1) {
      force.emplace_back(3 * i + 2, -1.0);
    }
  }

  FiniteElementAnalysis fea(mesh);
  fea.set_fixed_dof(bc);
  fea.set_forces(force);

  auto solution = fea.solve_system();

  // test that all bending and membrane stresses are 0.75
  auto stress = fea.compute_stress(solution);
  for (const auto& stress_manifold : stress[1])
    for (const auto& s : stress_manifold) {
      const double vm_stress = s[0] * s[0] + s[1] * s[1] - s[0] * s[1] + 3.0 * s[3] * s[3];
      REQUIRE(vm_stress == Catch::Approx(0.75).margin(1e-10));
    }
}

TEST_CASE("Bending curved shell", "[bend]") {

  std::cout << std::endl << "Partial cylinder" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/curved_shell_ribs*.ply", 0.1);
  mesh.setup_dof_numbering();

  const auto& manifold = mesh.manifolds[0];

  std::vector<std::pair<uint32_t, double>> force;
  std::vector<uint32_t> bc;
  for (size_t i = 0; i < manifold.Points().size(); i++) {
    const auto& pt = manifold.Points()[i];

    const bool isOnB1 = pt[0] > -0.001;

    if (isOnB1) {
      bc.push_back(3 * i + 0);
      bc.push_back(3 * i + 1);
      bc.push_back(3 * i + 2);
    }

    const bool isOnTopOrBot = pt[2] > 0.999 || pt[2] < -0.999;
    if (isOnTopOrBot) {
      bc.push_back(3 * i + 2);
    }
  }

  const double moment = 0.1;
  std::vector<std::pair<uint32_t, double>> motion_rotation;
  for (const auto& m : mesh.manifolds)
    for (size_t i = 0; i < m.Faces().size(); i++) {
      const auto& f         = m.Faces()[i];
      const bool pt1_isOnB1 = m.Points()[f[0]][0] > -0.001;
      const bool pt2_isOnB1 = m.Points()[f[1]][0] > -0.001;
      const bool pt3_isOnB1 = m.Points()[f[2]][0] > -0.001;

      const bool pt1_isOnB2 = m.Points()[f[0]][1] > -0.001;
      const bool pt2_isOnB2 = m.Points()[f[1]][1] > -0.001;
      const bool pt3_isOnB2 = m.Points()[f[2]][1] > -0.001;

      const bool pt1_isOnB3 = m.Points()[f[0]][2] > 0.999 || m.Points()[f[0]][2] < -0.999;
      const bool pt2_isOnB3 = m.Points()[f[1]][2] > 0.999 || m.Points()[f[1]][2] < -0.999;
      const bool pt3_isOnB3 = m.Points()[f[2]][2] > 0.999 || m.Points()[f[2]][2] < -0.999;

      const bool isClamped1 = (pt1_isOnB1 && pt2_isOnB1) || (pt1_isOnB3 && pt2_isOnB3);
      const bool isClamped2 = (pt2_isOnB1 && pt3_isOnB1) || (pt2_isOnB3 && pt3_isOnB3);
      const bool isClamped3 = (pt1_isOnB1 && pt3_isOnB1) || (pt1_isOnB3 && pt3_isOnB3);

      if (isClamped1) {
        const auto dof = m.EdgeDofNumber()[i][0];
        bc.push_back(dof);
      }
      if (isClamped2) {
        const auto dof = m.EdgeDofNumber()[i][1];
        bc.push_back(dof);
      }
      if (isClamped3) {
        const auto dof = m.EdgeDofNumber()[i][2];
        bc.push_back(dof);
      }

      if (pt1_isOnB2 && pt2_isOnB2) {
        const auto dof = m.EdgeDofNumber()[i][0];
        int n1         = m.UniqueNodeNumber()[f[0]];
        int n2         = m.UniqueNodeNumber()[f[1]];
        if (n1 > n2)
          force.emplace_back(dof, -moment);
        else
          force.emplace_back(dof, moment);
      }

      if (pt2_isOnB2 && pt3_isOnB2) {
        const auto dof = m.EdgeDofNumber()[i][1];
        int n2         = m.UniqueNodeNumber()[f[1]];
        int n3         = m.UniqueNodeNumber()[f[2]];
        if (n2 > n3)
          force.emplace_back(dof, -moment);
        else
          force.emplace_back(dof, moment);
      }

      if (pt1_isOnB2 && pt3_isOnB2) {
        const auto dof = m.EdgeDofNumber()[i][2];
        int n1         = m.UniqueNodeNumber()[f[0]];
        int n3         = m.UniqueNodeNumber()[f[2]];
        if (n3 > n1)
          force.emplace_back(dof, -moment);
        else
          force.emplace_back(dof, moment);
      }
    }

  FiniteElementAnalysis fea(mesh);
  fea.set_fixed_dof(bc);
  fea.set_forces(force);

  auto solution = fea.solve_system();
  auto stress   = fea.compute_stress(solution);

  // check that all membrane stresses are approx 0., small spurious stess is expected.
  for (const auto& stress_manifold : stress[0])
    for (const auto& s : stress_manifold) {
      const double vm_stress = s[0] * s[0] + s[1] * s[1] - s[0] * s[1] + 3.0 * s[3] * s[3];
      REQUIRE(vm_stress == Catch::Approx(0.0).margin(1e-5));
    }

  // check that rib bending stresses are approx 0., small spurious stress is expected.
  for (size_t i = 1; i < stress[1].size(); i++) {
    const auto& stress_manifold = stress[1][i];
    for (const auto& s : stress_manifold) {
      const double vm_stress = s[0] * s[0] + s[1] * s[1] - s[0] * s[1] + 3.0 * s[3] * s[3];
      REQUIRE(vm_stress == Catch::Approx(0.0).margin(1e-5));
    }
  }
}