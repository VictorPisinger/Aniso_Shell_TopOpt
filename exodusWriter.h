#pragma once

#include <string>
#include <vector>

#include "eigen3/Eigen/Dense"
#include "mesh.h"

class ExodusWriter {

public:
  explicit ExodusWriter(const std::string fname, const Mesh& mesh, std::vector<std::string> pointFieldNames,
                        std::vector<std::string> elementFieldNames);

  void writeNodeField(const std::vector<std::vector<double>>& value, int fieldNo) const;

  void writeThickness(int fieldNo) const;

  void writeDisplacement(const Eigen::VectorXd& displacement, int startField) const;

  void writeTensor(const std::vector<std::vector<std::array<double, 6>>> tensor, int startField) const;

  void writeElementField(const std::vector<std::vector<double>> field, int field_number) const;

  void incrementTimestep() { timestep++; }

private:
  std::string filename;
  const Mesh& mesh_;
  int timestep = 1;
};