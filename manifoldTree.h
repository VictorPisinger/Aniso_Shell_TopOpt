#pragma once

#include <nanoflann.hpp>

// Adaptor class to interface with kdtree library nanoflann
struct ManifoldAdaptor {
  using Derived = std::vector<std::array<double, 3>>;

  using coord_t = double;
  const Derived& obj; //!< A const ref to the data set origin

  /// The constructor that sets the data set source
  ManifoldAdaptor(const Derived& obj_) : obj(obj_) {}

  /// CRTP helper method
  inline const Derived& derived() const { return obj; }

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return derived().size(); }

  inline coord_t kdtree_get_pt(const size_t idx, const size_t dim) const { return derived()[idx][dim]; }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }

  // Returns the distance between the vector "p1[0:size-1]" and the data point
  // with index "idx_p2" stored in the class:
  inline coord_t kdtree_distance(const coord_t* p1, const size_t idx_p2, size_t /*size*/) const {
    const coord_t d0 = p1[0] - derived()[idx_p2][0];
    const coord_t d1 = p1[1] - derived()[idx_p2][1];
    const coord_t d2 = p1[2] - derived()[idx_p2][2];
    return d0 * d0 + d1 * d1 + d2 * d2;
  }

}; // end of ManifoldAdaptor

// setup kd tree for fast spatial lookup on list of all points
using ManifoldTree =
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, ManifoldAdaptor>, ManifoldAdaptor, 3>;