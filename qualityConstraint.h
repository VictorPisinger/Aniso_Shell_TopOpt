#pragma once

#include "manifold.h"
#include "mesh.h"

class QualityConstraint {

public:
  QualityConstraint(const Mesh& mesh, const double allowedAspectRatio = 3.0)
      : mesh_(mesh), allowedAspectRatio_(allowedAspectRatio){};

  double compute_AR_true() const;
  double compute_AR_value() const;
  std::vector<double> compute_AR_shape_gradient() const;

private:
  const Mesh& mesh_;
  const double allowedAspectRatio_;
  const double p_ = 20.0;
};