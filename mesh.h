#pragma once

#include <vector>

#include "manifold.h"

class Mesh {

public:
  double minimum_edge_length() const;

  uint32_t number_of_triangles() const;

  void read_ply_files(const std::string fileBaseName, const double default_thickness = 0.0);

  void write_ply_files(const std::string fileBaseName) const;

  void setup_dof_numbering();

  void move_nodes(const std::vector<double> node_movement);

  void set_nodes(const std::vector<double> node_movement);

  std::vector<Manifold> manifolds;
  uint32_t numberOfUniqueNodes = 0;
  uint32_t numberOfDofs        = 0;
};