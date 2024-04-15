#include "mesh.h"

#include <algorithm>
#include <fstream>
#include <tuple>
#include <unordered_map>

#include <iostream>

#include "manifoldTree.h"

double Mesh::minimum_edge_length() const {
  double minEdgeLength = std::numeric_limits<double>::max();

  for (const auto& manifold : manifolds) {
    minEdgeLength = std::min(minEdgeLength, manifold.minimum_edge_length());
  }

  return minEdgeLength;
}

uint32_t Mesh::number_of_triangles() const {
  uint32_t num = 0;
  for (const auto& manifold : manifolds) {
    num += manifold.Faces().size();
  }
  return num;
}

void Mesh::read_ply_files(const std::string fileBaseName, const double default_thickness) {

  // setup variables for filename construction.
  auto pos  = fileBaseName.find('*');
  auto pre  = fileBaseName.substr(0, pos);
  auto post = fileBaseName.substr(pos + 1, fileBaseName.size() - pos - 1);

  for (size_t i = 0;; i++) {

    std::string filename = pre + std::to_string(i) + post;
    manifolds.emplace_back();

    try {
      manifolds.back().read_ply(filename, default_thickness);
    } catch (const std::exception& e) {
      // if file fails reading remove manifold and stop trying new files.
      manifolds.pop_back();
      break;
    }
  }

  if (manifolds.size() == 0) {
    std::cout << "WARNING, no manifolds after reading " << fileBaseName << std::endl
              << "Either the files do not exist, or they are empty." << std::endl;
  }
}

void Mesh::write_ply_files(const std::string fileBaseName) const {

  // setup variables for filename construction.
  auto pos  = fileBaseName.find('*');
  auto pre  = fileBaseName.substr(0, pos);
  auto post = fileBaseName.substr(pos + 1, fileBaseName.size() - pos - 1);

  for (size_t i = 0; i < manifolds.size(); i++) {

    std::string filename = pre + std::to_string(i) + post;
    manifolds[i].write_ply(filename);
  }
}

void Mesh::setup_dof_numbering() {

  // setup global point-cloud, and cloud to manifold-points list.
  std::vector<std::array<double, 3>> allPoints;
  std::vector<std::pair<size_t, size_t>> pointIndex;

  size_t numberOfPoints = 0;
  for (const auto& m : manifolds) {
    numberOfPoints += m.Points().size();
  }
  allPoints.reserve(numberOfPoints);
  pointIndex.reserve(numberOfPoints);

  for (size_t i = 0; i < manifolds.size(); i++) {
    for (size_t j = 0; j < manifolds[i].Points().size(); j++) {
      allPoints.emplace_back(manifolds[i].Points()[j]);
      pointIndex.emplace_back(i, j);
    }
  }

  ManifoldAdaptor adaptor(allPoints);
  ManifoldTree kdTree(3 /*dim*/, adaptor, 50 /* max leaf */);
  kdTree.buildIndex();

  // setup a squared radius that ensures that it is the same points
  const double minEdge            = 0.5 * this->minimum_edge_length();
  const double geometricTolerance = 0.5 * minEdge * minEdge;

  // setup containers and structs for kd  tree library
  nanoflann::SearchParams kdTreeParams;
  std::vector<std::pair<size_t, double>> searchResults;
  searchResults.reserve(10);

  // start by marking all unique node numbers as invalid.
  const uint32_t invalidIndex = std::numeric_limits<uint32_t>::max();
  for (auto& m : manifolds) {
    m.uniqueNodeNumber.clear();
    m.uniqueNodeNumber.resize(m.Points().size(), invalidIndex);
  }

  // go through all nodes, find all nodes at 'same' position, and set them to
  // the same unique node number. If a node is missed due to the approximative
  // nature of the KD tree, we have big trouble.
  uint32_t currentIndex = 0;
  for (const auto& m : manifolds) {
    for (size_t j = 0; j < m.Points().size(); j++) {

      if (m.uniqueNodeNumber[j] == invalidIndex) {
        const auto& pt = m.Points()[j];

        kdTree.radiusSearch(pt.data(), geometricTolerance, searchResults, kdTreeParams);

        for (auto& match : searchResults) {
          const auto& idx                                   = pointIndex[match.first];
          manifolds[idx.first].uniqueNodeNumber[idx.second] = currentIndex;
        }
        currentIndex++;
      }
    }
  }

  // save the total number of unique nodes for future reference.
  this->numberOfUniqueNodes = currentIndex;

  // setup edge dofs

  // Hash function
  struct EdgeHash {
    size_t operator()(const std::pair<uint32_t, uint32_t>& x) const { return x.first + x.second * N; }
    const uint32_t N = 1e6;
  };
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, EdgeHash> edges;

  for (const auto& m : manifolds) {
    for (const auto& f : m.faces) {
      const auto n1 = m.uniqueNodeNumber[f[0]];
      const auto n2 = m.uniqueNodeNumber[f[1]];
      const auto n3 = m.uniqueNodeNumber[f[2]];

      edges.insert({std::make_pair(std::min(n1, n2), std::max(n1, n2)), 0});
      edges.insert({std::make_pair(std::min(n2, n3), std::max(n2, n3)), 0});
      edges.insert({std::make_pair(std::min(n1, n3), std::max(n1, n3)), 0});
    }
  }

  // set edge
  uint32_t dof_count = 3 * this->numberOfUniqueNodes;
  for (auto& edge : edges) {
    edge.second = dof_count;
    dof_count++;
  }

  this->numberOfDofs = dof_count;

  for (auto& m : manifolds) {
    m.edgeDofNumber.resize(m.faces.size());

    for (size_t j = 0; j < m.faces.size(); j++) {
      const auto& f = m.faces[j];
      const auto n1 = m.uniqueNodeNumber[f[0]];
      const auto n2 = m.uniqueNodeNumber[f[1]];
      const auto n3 = m.uniqueNodeNumber[f[2]];

      m.edgeDofNumber[j][0] = edges.at(std::make_pair(std::min(n1, n2), std::max(n1, n2)));
      m.edgeDofNumber[j][1] = edges.at(std::make_pair(std::min(n2, n3), std::max(n2, n3)));
      m.edgeDofNumber[j][2] = edges.at(std::make_pair(std::min(n1, n3), std::max(n1, n3)));
    }
  }
}

void Mesh::move_nodes(const std::vector<double> node_movement) {
  for (auto& manifold : manifolds) {
    manifold.move_nodes(node_movement);
  }
}

void Mesh::set_nodes(const std::vector<double> node_movement) {
  for (auto& manifold : manifolds) {
    manifold.set_nodes(node_movement);
  }
}
