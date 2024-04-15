#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "mesh.h"

class Options {

public:
  Options(int argc, char* argv[]);

  std::vector<uint32_t> get_boundary_dofs(const Mesh& mesh) const;

  std::vector<std::pair<uint32_t, double>> get_forces(const Mesh& mesh) const;

  std::vector<std::pair<size_t, double>> get_pressure_manifolds() const;

  std::array<std::shared_ptr<std::vector<uint32_t>>, 3> get_non_design_nodes(const Mesh& mesh) const;

  template <typename T>
  T get(const std::string key, T default_value) const {

    // if key exists in yaml, overwrite the default value
    if (options[key]) {
      default_value = options[key].as<T>();
    }

    return default_value;
  }

  bool exists(const std::string key) const { return static_cast<bool>(options[key]); }

  void get_movement_bounds(const Mesh& mesh, const double min_val, const double max_val, std::vector<double>& move_min,
                           std::vector<double>& move_max) const;

private:
  YAML::Node rootnode;
  YAML::Node options;
};