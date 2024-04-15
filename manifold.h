#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

class Manifold {

public:
  Manifold() = default;

  Manifold(std::vector<std::array<double, 3>> pt, std::vector<std::array<uint32_t, 3>> facets, const double thk)
      : initial_coordinates(pt), current_coordinates(pt), faces(facets) {
    thickness.resize(pt.size(), thk);
  }

  friend class Mesh;

  double minimum_edge_length() const;

  void read_ply(const std::string& filename, const double default_thickness = 0.0);

  void write_ply(const std::string& fileName) const;

  void write_ply_with_displacement(const std::string& fileName, const Eigen::VectorXd disp) const;

  void move_nodes(const std::vector<double> node_movement);

  void set_nodes(const std::vector<double> node_movement);

  const std::vector<std::array<double, 3>>& Points() const { return current_coordinates; }
  const std::vector<std::array<double, 3>>& InitialPoints() const { return initial_coordinates; }
  std::vector<double>& ThicknessMutable() { return thickness; }
  const std::vector<double>& Thickness() const { return thickness; }
  const std::vector<uint32_t>& UniqueNodeNumber() const { return this->uniqueNodeNumber; }

  const std::vector<std::array<uint32_t, 3>>& Faces() const { return this->faces; }
  const std::vector<std::array<uint32_t, 3>>& EdgeDofNumber() const { return this->edgeDofNumber; }

private:
  // nodally defined vairables
  std::vector<std::array<double, 3>> initial_coordinates;
  std::vector<std::array<double, 3>> current_coordinates;
  std::vector<double> thickness;
  std::vector<uint32_t> uniqueNodeNumber;

  // faces
  std::vector<std::array<uint32_t, 3>> faces;
  std::vector<std::array<uint32_t, 3>> edgeDofNumber;
};