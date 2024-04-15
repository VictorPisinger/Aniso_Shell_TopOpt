#pragma once

#include "manifold.h"
#include "mesh.h"

class VolumeConstraint {

public:
  VolumeConstraint(const Mesh& mesh, const double fractionOfInitialVolume = 1.0)
      : mesh_(mesh), allowedVolume_(fractionOfInitialVolume * compute_volume(mesh)){};

  static double compute_volume(const Manifold& manifold);
  static double compute_volume(const Mesh& mesh);
  double compute_volume() const { return compute_volume(mesh_); };

  double compute_value() const;
  std::vector<std::vector<double>> compute_thickness_gradient() const;
  std::vector<double> compute_shape_gradient() const;

  void set_allowed_volume(const double allowed_volume) { allowedVolume_ = allowed_volume; };

private:
  const Mesh& mesh_;
  double allowedVolume_;
};