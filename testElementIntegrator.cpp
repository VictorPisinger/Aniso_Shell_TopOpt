#include "catch2/catch_test_macros.hpp"

#include "catch2/catch_approx.hpp"

#include "elementIntegrator.h"
#include "shellElement.h"

#include <iostream>

// Shared fixtures for all test
ElementIntegrator ei;
constexpr double tolerance = 1e-8;

TEST_CASE("Compute New Coordinate system", "[Coord]") {

  std::cout << std::endl << "coordinate system" << std::endl;

  Eigen::Vector3d p1 = {0, 2, 3};
  Eigen::Vector3d p2 = {1, 0.5, 3};
  Eigen::Vector3d p3 = {3, 0, 0.75};

  auto basis = Shell::compute_local_basis<double>({p1, p2, p3});

  // check that basis is orthonormal
  std::cout << "dots " << basis[0].dot(basis[1]) << " " << basis[0].dot(basis[2]) << " " << basis[1].dot(basis[2])
            << " " << std::endl;

  REQUIRE(basis[0].norm() == Catch::Approx(1.0).margin(tolerance));
  REQUIRE(basis[1].norm() == Catch::Approx(1.0).margin(tolerance));
  REQUIRE(basis[2].norm() == Catch::Approx(1.0).margin(tolerance));
  REQUIRE(basis[0].dot(basis[1]) == Catch::Approx(0.0).margin(tolerance));
  REQUIRE(basis[0].dot(basis[2]) == Catch::Approx(0.0).margin(tolerance));
  REQUIRE(basis[1].dot(basis[2]) == Catch::Approx(0.0).margin(tolerance));

  // check that third coordinate is constant when the corners are transformed to
  // the new system.
  const auto T = Shell::compute_rotation_matrix(basis);

  Eigen::Vector3d p1_new = T * p1;
  Eigen::Vector3d p2_new = T * p2;
  Eigen::Vector3d p3_new = T * p3;

  std::cout << "transformed p1 " << p1.transpose() << " to " << p1_new.transpose() << std::endl;
  std::cout << "transformed p2 " << p2.transpose() << " to " << p2_new.transpose() << std::endl;
  std::cout << "transformed p3 " << p3.transpose() << " to " << p3_new.transpose() << std::endl;

  REQUIRE(p2_new(2) == Catch::Approx(p1_new(2)).margin(tolerance));
  REQUIRE(p3_new(2) == Catch::Approx(p1_new(2)).margin(tolerance));

  // check that the tranpse works as inverse
  Eigen::Vector3d p1_back = T.transpose() * p1_new;
  Eigen::Vector3d p2_back = T.transpose() * p2_new;
  Eigen::Vector3d p3_back = T.transpose() * p3_new;

  for (int i = 0; i < 3; i++) {
    REQUIRE(p1_back(i) == Catch::Approx(p1(i)).margin(tolerance));
    REQUIRE(p2_back(i) == Catch::Approx(p2(i)).margin(tolerance));
    REQUIRE(p3_back(i) == Catch::Approx(p3(i)).margin(tolerance));
  }
}

TEST_CASE("Compute inplane coefficients", "[inplane]") {

  std::cout << std::endl << "inplane coefficients" << std::endl;

  Eigen::Vector3d p1 = {0, 2, 3};
  Eigen::Vector3d p2 = {1, 0.5, 3};
  Eigen::Vector3d p3 = {3, 0, 0.75};
  auto basis         = Shell::compute_local_basis<double>({p1, p2, p3});

  std::cout << "e1: " << basis[0].transpose() << std::endl;
  std::cout << "e2: " << basis[1].transpose() << std::endl;
  std::cout << "e3: " << basis[2].transpose() << std::endl;

  const auto coeffs = Shell::compute_inplane_coefficients<double>({p1, p2, p3}, basis);

  std::cout << "a: " << coeffs[0].transpose() << std::endl;
  std::cout << "b: " << coeffs[1].transpose() << std::endl;

  const Eigen::Vector3d s1 = p3 - p2;
  const Eigen::Vector3d s2 = p1 - p3;
  const Eigen::Vector3d s3 = p2 - p1;

  const Eigen::Vector3d ss1 = coeffs[0](0) * basis[0] - coeffs[1](0) * basis[1];
  const Eigen::Vector3d ss2 = coeffs[0](1) * basis[0] - coeffs[1](1) * basis[1];
  const Eigen::Vector3d ss3 = coeffs[0](2) * basis[0] - coeffs[1](2) * basis[1];

  std::cout << "s1: " << s1.transpose() << std::endl;
  std::cout << "ss1: " << ss1.transpose() << std::endl << std::endl;

  std::cout << "s2: " << s2.transpose() << std::endl;
  std::cout << "ss2: " << ss2.transpose() << std::endl << std::endl;

  std::cout << "s3: " << s3.transpose() << std::endl;
  std::cout << "ss3: " << ss3.transpose() << std::endl << std::endl;

  for (int i = 0; i < 2; i++) {
    REQUIRE(s1(i) == Catch::Approx(ss1(i)).margin(tolerance));
    REQUIRE(s2(i) == Catch::Approx(ss2(i)).margin(tolerance));
    REQUIRE(s3(i) == Catch::Approx(ss3(i)).margin(tolerance));
  }
}

TEST_CASE("Compute triangle area", "[area]") {

  Eigen::Vector3d p1 = {0, 0, 0};
  Eigen::Vector3d p2 = {0, 0, 1};
  Eigen::Vector3d p3 = {1, 0, 0};
  const double area  = Shell::compute_area<double>({p1, p2, p3});
  REQUIRE(area == Catch::Approx(0.5).margin(tolerance));

  p2                 = 10 * p2;
  p3                 = 10 * p3;
  const double area2 = Shell::compute_area<double>({p1, p2, p3});
  REQUIRE(area2 == Catch::Approx(10 * 10 * 0.5).margin(tolerance));
}

TEST_CASE("Compute bending stiffness", "[bending]") {

  Eigen::Vector3d p1 = {0, 0, 0};
  Eigen::Vector3d p2 = {0, 0, 1};
  Eigen::Vector3d p3 = {1, 0, 0};
  auto basis         = Shell::compute_local_basis<double>({p1, p2, p3});

  const auto T           = Shell::compute_rotation_matrix(basis);
  Eigen::Vector3d p1_new = T * p1;
  Eigen::Vector3d p2_new = T * p2;
  Eigen::Vector3d p3_new = T * p3;

  auto kb = Shell::compute_bending_stiffness({p1_new, p2_new, p3_new}, {false, false, false}, 0.1, 1.0, 0.3);

  std::cout << std::endl << "bending stiffness" << std::endl << kb << std::endl;

  // constrol that bending stiffness is symmetric
  auto kbdiff = kb.transpose() - kb;
  REQUIRE(kbdiff.norm() == Catch::Approx(0.0).margin(tolerance));
}

TEST_CASE("Compute membrane stiffness", "[membrane]") {

  Eigen::Vector3d p1 = {0, 2, 3};
  Eigen::Vector3d p2 = {1, 0.5, 3};
  Eigen::Vector3d p3 = {3, 0, 0.75};
  auto basis         = Shell::compute_local_basis<double>({p1, p2, p3});
  auto km            = Shell::compute_membrane_stiffness({p1, p2, p3}, basis, 0.1, 1.0, 0.3);

  std::cout << std::endl << "membrane stiffness" << std::endl << km << std::endl;

  // constrol that membrane stiffness is symmetric
  auto kmdiff = km.transpose() - km;
  REQUIRE(kmdiff.norm() == Catch::Approx(0.0).margin(tolerance));
}

TEST_CASE("Compute local stiffness", "[ke]") {

  Eigen::Vector3d p1 = {0, 0, 0};
  Eigen::Vector3d p2 = {1, 0, 0};
  Eigen::Vector3d p3 = {0, 1, 0};
  auto ke            = ei.compute_element_stiffness({p1, p2, p3}, {false, false, false}, 0.1);

  std::cout << std::endl << "local stiffness" << std::endl << ke << std::endl;

  // constrol that membrane stiffness is symmetric
  auto kediff = ke.transpose() - ke;
  REQUIRE(kediff.norm() == Catch::Approx(0.0).margin(tolerance));

  // check that u,v and w,rot are not coupled
  for (int i = 0; i < 6; i++) {
    for (int j = 6; j < 12; j++) {
      REQUIRE(ke(i, j) == Catch::Approx(0.0).margin(tolerance));
    }
  }

  auto ke_switched = ei.compute_element_stiffness({p1, p2, p3}, {false, true, false}, 0.1);

  auto keswitchdiff = ke_switched - ke;
  std::cout << std::endl << "local stiffness difference" << std::endl << keswitchdiff << std::endl;
}

TEST_CASE("local stiffness rotation", "[ke_rot]") {

  Eigen::Vector3d p1 = {0, 0, 0};
  Eigen::Vector3d p2 = {1, 0, 0};
  Eigen::Vector3d p3 = {0, 0, 1};
  auto ke            = ei.compute_element_stiffness({p1, p2, p3}, {false, false, false}, 0.1);

  std::cout << std::endl << "local stiffness" << std::endl << ke << std::endl;

  // constrol that membrane stiffness is symmetric
  auto kediff = ke.transpose() - ke;
  REQUIRE(kediff.norm() == Catch::Approx(0.0).margin(tolerance));

  // check that u and v are not coupled
  for (int i = 0; i < 3; i++) {
    for (int j = 3; j < 6; j++) {
      REQUIRE(ke(i, j) == Catch::Approx(0.0).margin(tolerance));
    }
  }

  // check that w and v are not coupled
  for (int i = 6; i < 9; i++) {
    for (int j = 3; j < 6; j++) {
      REQUIRE(ke(i, j) == Catch::Approx(0.0).margin(tolerance));
    }
  }
}

TEST_CASE("Compute local energy", "[energy]") {
  std::cout << std::endl << "local energy" << std::endl;

  Eigen::Vector3d p1 = {0, 0, 0};
  Eigen::Vector3d p2 = {1, 0, 0};
  Eigen::Vector3d p3 = {0, 0, 1};

  Eigen::Vector<double, 12> dispRigid = {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0};

  auto energy = ei.compute_bending_and_membrane_compliance({p1, p2, p3}, {false, false, false}, 0.1, dispRigid);

  std::cout << "Rigid body modes " << energy[0] << " " << energy[1] << std::endl;

  // constrol that rigid body modes have 0 energy
  REQUIRE(energy[0] == Catch::Approx(0.0).margin(tolerance));
  REQUIRE(energy[1] == Catch::Approx(0.0).margin(tolerance));
}

TEST_CASE("Compute surface operators", "[filter]") {
  std::cout << std::endl << "surface operators" << std::endl;

  // Eigen::Vector3d p1 = {0, 2, 3};
  // Eigen::Vector3d p2 = {1, 0.5, 3};
  // Eigen::Vector3d p3 = {3, 0, 0.75};

  Eigen::Vector3d p1 = {0, 0, 0};
  Eigen::Vector3d p2 = {2, 0, 0};
  Eigen::Vector3d p3 = {0, 0, 2};

  auto m = Shell::compute_geometry_mass<double>({p1, p2, p3});
  auto k = Shell::compute_geometry_laplace<double>({p1, p2, p3});

  std::cout << "m " << std::endl << m << std::endl << "k " << std::endl << k << std::endl;
}
