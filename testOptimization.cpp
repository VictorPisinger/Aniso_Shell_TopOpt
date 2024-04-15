#include "catch2/catch_test_macros.hpp"

#include "catch2/catch_approx.hpp"

#include "exodusWriter.h"
#include "finiteElementAnalysis.h"
#include "mesh.h"
#include "pdeFilter.h"
#include "qualityConstraint.h"
#include "volumeConstraint.h"

#include <iostream>

TEST_CASE("Test filter radius and kernel 04") {
  std::cout << "Filter radius" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/plate_04_*.ply", 0.1);
  mesh.setup_dof_numbering();

  std::vector<uint32_t> bcNodes;
  std::vector<double> input_field(mesh.numberOfUniqueNodes * 3, 0.0);

  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1     = pt[0] < -0.499;
    const bool isOnB2     = pt[0] > 0.499;
    const bool isOnB3     = pt[1] < -0.499;
    const bool isOnB4     = pt[1] > 0.499;
    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4) {
      bcNodes.push_back(3 * i + 0);
      bcNodes.push_back(3 * i + 1);
      bcNodes.push_back(3 * i + 2);
    }

    if (isInCenter) {
      input_field[3 * i + 2] = 1.0;
    }
  }

  PdeFilter filter(mesh, 0.3);

  const auto filtered_field = filter.forward(input_field);

  ExodusWriter writer("filtertest_04.e", mesh, {"ix", "iy", "iz", "ox", "oy", "oz"}, {});
  writer.writeDisplacement(Eigen::VectorXd::Map(&input_field[0], input_field.size()), 0);
  writer.writeDisplacement(Eigen::VectorXd::Map(&filtered_field[0], filtered_field.size()), 3);
}

TEST_CASE("Test filter radius and kernel 02") {
  std::cout << "Filter radius" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/plate_02_*.ply", 0.1);
  mesh.setup_dof_numbering();

  std::vector<uint32_t> bcNodes;
  std::vector<double> input_field(mesh.numberOfUniqueNodes * 3, 0.0);

  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1     = pt[0] < -0.499;
    const bool isOnB2     = pt[0] > 0.499;
    const bool isOnB3     = pt[1] < -0.499;
    const bool isOnB4     = pt[1] > 0.499;
    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4) {
      bcNodes.push_back(3 * i + 0);
      bcNodes.push_back(3 * i + 1);
      bcNodes.push_back(3 * i + 2);
    }

    if (isInCenter) {
      input_field[3 * i + 2] = 1.0;
    }
  }

  PdeFilter filter(mesh, 0.3, {});

  const auto filtered_field = filter.forward(input_field);

  ExodusWriter writer("filtertest_02.e", mesh, {"ix", "iy", "iz", "ox", "oy", "oz"}, {});
  writer.writeDisplacement(Eigen::VectorXd::Map(&input_field[0], input_field.size()), 0);
  writer.writeDisplacement(Eigen::VectorXd::Map(&filtered_field[0], filtered_field.size()), 3);
}

TEST_CASE("Test filter radius and kernel 01") {
  std::cout << "Filter radius" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/plate_01_*.ply", 0.1);
  mesh.setup_dof_numbering();

  std::vector<uint32_t> bcNodes;
  std::vector<double> input_field(mesh.numberOfUniqueNodes * 3, 0.0);

  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1     = pt[0] < -0.499;
    const bool isOnB2     = pt[0] > 0.499;
    const bool isOnB3     = pt[1] < -0.499;
    const bool isOnB4     = pt[1] > 0.499;
    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4) {
      bcNodes.push_back(3 * i + 0);
      bcNodes.push_back(3 * i + 1);
      bcNodes.push_back(3 * i + 2);
    }

    if (isInCenter) {
      input_field[3 * i + 2] = 1.0;
    }
  }

  PdeFilter filter(mesh, 0.3, {});

  const auto filtered_field = filter.forward(input_field);

  ExodusWriter writer("filtertest_01.e", mesh, {"ix", "iy", "iz", "ox", "oy", "oz"}, {});
  writer.writeDisplacement(Eigen::VectorXd::Map(&input_field[0], input_field.size()), 0);
  writer.writeDisplacement(Eigen::VectorXd::Map(&filtered_field[0], filtered_field.size()), 3);
}

TEST_CASE("Test filter radius and kernel 005") {
  std::cout << "Filter radius" << std::endl;

  Mesh mesh;
  mesh.read_ply_files("../testMeshes/plate_005_*.ply", 0.1);
  mesh.setup_dof_numbering();

  std::vector<uint32_t> bcNodes;
  std::vector<double> input_field(mesh.numberOfUniqueNodes * 3, 0.0);

  for (size_t i = 0; i < mesh.manifolds[0].Points().size(); i++) {
    const auto& pt = mesh.manifolds[0].Points()[i];

    const bool isOnB1     = pt[0] < -0.499;
    const bool isOnB2     = pt[0] > 0.499;
    const bool isOnB3     = pt[1] < -0.499;
    const bool isOnB4     = pt[1] > 0.499;
    const bool isInCenter = (pt[0] < 0.001) && (pt[0] > -0.001) && (pt[1] < 0.001) && (pt[1] > -0.001);

    if (isOnB1 || isOnB2 || isOnB3 || isOnB4) {
      bcNodes.push_back(3 * i + 0);
      bcNodes.push_back(3 * i + 1);
      bcNodes.push_back(3 * i + 2);
    }

    if (isInCenter) {
      input_field[3 * i + 2] = 1.0;
    }
  }

  PdeFilter filter(mesh, 0.3, {});

  const auto filtered_field = filter.forward(input_field);

  ExodusWriter writer("filtertest_005.e", mesh, {"ix", "iy", "iz", "ox", "oy", "oz"}, {});
  writer.writeDisplacement(Eigen::VectorXd::Map(&input_field[0], input_field.size()), 0);
  writer.writeDisplacement(Eigen::VectorXd::Map(&filtered_field[0], filtered_field.size()), 3);
}