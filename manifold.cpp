#include "manifold.h"

#include <fstream>
#include <limits>
#include <sstream>

#include "utilities.h"

double Manifold::minimum_edge_length() const {
  double minEdgeLength = std::numeric_limits<double>::max();

  for (const auto& tri : faces) {
    const auto& p1 = current_coordinates[tri[0]];
    const auto& p2 = current_coordinates[tri[1]];
    const auto& p3 = current_coordinates[tri[2]];

    minEdgeLength = std::min(minEdgeLength, distance(p1, p2));
    minEdgeLength = std::min(minEdgeLength, distance(p2, p3));
    minEdgeLength = std::min(minEdgeLength, distance(p3, p1));
  }

  return minEdgeLength;
}

/**
 * @brief Writes the contents of this manifold to a ply file. Again, only
 * trinagles are supported for the
 *
 * @param fileName
 */
void Manifold::write_ply(const std::string& fileName) const {

  if (this->current_coordinates.size() < 1)
    return;

  // https://en.wikipedia.org/wiki/PLY_(file_format)

  std::ofstream file;
  file.open(fileName, std::ios::out);

  // throws an exception if file is not found
  file.exceptions(file.failbit);

  const size_t Np = this->current_coordinates.size();

  file << "ply\n"
       << "format ascii 1.0\n"
       << "element vertex " << Np << "\n"
       << "property float x\n"
       << "property float y\n"
       << "property float z\n"
       << "property uchar red\n"
       << "property uchar green\n"
       << "property uchar blue\n"
       << "element face " << faces.size() << "\n"
       << "property list uchar int vertex_index\n"
       << "end_header\n";

  for (size_t i = 0; i < Np; i++) {
    auto& p = this->current_coordinates[i];
    int c   = 255;
    if (this->thickness.size() != 0)
      c = int(255.0 * this->thickness[i]);

    if (this->uniqueNodeNumber.size() != 0)
      c = int(this->uniqueNodeNumber[i]);
    file << p[0] << " " << p[1] << " " << p[2] << " ";
    file << c << " " << c << " " << c << "\n";
  }

  for (const auto& f : faces)
    file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

  file.close();
}

void Manifold::write_ply_with_displacement(const std::string& fileName, const Eigen::VectorXd disp) const {

  if (this->current_coordinates.size() < 1)
    return;

  // https://en.wikipedia.org/wiki/PLY_(file_format)

  std::ofstream file;
  file.open(fileName, std::ios::out);

  // throws an exception if file is not found
  file.exceptions(file.failbit);

  const size_t Np = this->current_coordinates.size();

  file << "ply\n"
       << "format ascii 1.0\n"
       << "element vertex " << Np << "\n"
       << "property float x\n"
       << "property float y\n"
       << "property float z\n"
       << "property uchar red\n"
       << "property uchar green\n"
       << "property uchar blue\n"
       << "element face " << faces.size() << "\n"
       << "property list uchar int vertex_index\n"
       << "end_header\n";

  double maxux = 0.0;
  double maxuy = 0.0;
  double maxuz = 0.0;
  double minux = 0.0;
  double minuy = 0.0;
  double minuz = 0.0;
  for (size_t i = 0; i < Np; i++) {
    maxux = std::max(maxux, disp(3 * this->uniqueNodeNumber[i] + 0));
    maxuy = std::max(maxuy, disp(3 * this->uniqueNodeNumber[i] + 1));
    maxuz = std::max(maxuz, disp(3 * this->uniqueNodeNumber[i] + 2));
    minux = std::min(minux, disp(3 * this->uniqueNodeNumber[i] + 0));
    minuy = std::min(minuy, disp(3 * this->uniqueNodeNumber[i] + 1));
    minuz = std::min(minuz, disp(3 * this->uniqueNodeNumber[i] + 2));
  }

  for (size_t i = 0; i < Np; i++) {
    auto& p = this->current_coordinates[i];

    const double ux = disp(3 * this->uniqueNodeNumber[i] + 0);
    const double uy = disp(3 * this->uniqueNodeNumber[i] + 1);
    const double uz = disp(3 * this->uniqueNodeNumber[i] + 2);

    const int ux_c = int(255 * (ux - minux) / (maxux - minux));
    const int uy_c = int(255 * (uy - minuy) / (maxuy - minuy));
    const int uz_c = int(255 * (uz - minuz) / (maxuz - minuz));

    file << p[0] << " " << p[1] << " " << p[2] << " ";
    file << ux_c << " " << uy_c << " " << uz_c << "\n";
  }

  for (const auto& f : faces)
    file << "3 " << f[0] << " " << f[1] << " " << f[2] << "\n";

  file.close();
}

// Function used to import pointclouds as .ply files
void Manifold::read_ply(const std::string& fileName, const double default_thickness) {

  // https://en.wikipedia.org/wiki/PLY_(file_format)

  // Assumptions:
  // - file exits

  std::ifstream file(fileName);

  // throws an exception if file is not found
  file.exceptions(file.failbit);

  // if (!file.good())
  //   exit(EXIT_FAILURE);

  // File reading variables
  std::string line, cell;
  std::stringstream lineStream(line);
  std::vector<double> float_values;
  std::vector<uint32_t> uint_values;
  std::vector<std::string> header;

  // Meta information
  std::vector<std::string> element_types;
  std::vector<int> element_amount;

  getline(file, line);
  if (line.compare("ply") != 0)
    exit(EXIT_FAILURE);

  // Read the header for information on file contents
  while (line.compare("end_header") != 0) {

    // File inputs
    getline(file, line);
    lineStream = std::stringstream(line);
    header.clear();
    while (getline(lineStream, cell, ' '))
      header.push_back(cell);

    // Read elements
    if (header.size() != 3)
      continue;
    if (header[0].compare("element") == 0) {
      element_types.push_back(header[1]);
      element_amount.push_back(stoi(header[2]));
    }
  }

  for (size_t e = 0; e < element_types.size(); e++) {
    std::string e_type = element_types[e];
    size_t e_amount    = element_amount[e];

    // Read in the vertex elements
    if (e_type.compare("vertex") == 0) {

      // Prepare variables
      this->current_coordinates.resize(e_amount);
      this->initial_coordinates.resize(e_amount);
      this->thickness.resize(e_amount, default_thickness);

      // Read in the information from the file
      for (size_t vertex_nr = 0; vertex_nr < e_amount; vertex_nr++) {

        // Get values from file
        getline(file, line);
        lineStream = std::stringstream(line);
        float_values.clear();
        while (getline(lineStream, cell, ' ')) {
          try {
            float_values.push_back(stod(cell));
          } catch (const std::out_of_range& oor) {
            float_values.push_back(0.0);
          }
        }

        // Read vertex positions
        if (float_values.size() < 3)
          continue;
        this->current_coordinates[vertex_nr] = {float_values[0], float_values[1], float_values[2]};
        this->initial_coordinates[vertex_nr] = {float_values[0], float_values[1], float_values[2]};

        // Read vertex normals
        if (float_values.size() < 6)
          continue;

        if (float_values.size() < 7)
          continue;
        this->thickness[vertex_nr] = float_values[6] / 255.0;
      }
    }

    else if (e_type.compare("face") == 0) {

      // Prepare variables
      this->faces.resize(e_amount);

      // Read in the information from the file
      for (size_t face_nr = 0; face_nr < e_amount; face_nr++) {
        // Get values from file
        getline(file, line);
        lineStream = std::stringstream(line);
        uint_values.clear();
        while (getline(lineStream, cell, ' ')) {
          try {
            uint_values.push_back(stod(cell));
          } catch (const std::out_of_range& oor) {
            uint_values.push_back(0);
          }
        }

        if (uint_values.size() < 4)
          continue;

        // only works for triangular faces
        if (uint_values[0] != 3)
          continue;

        this->faces[face_nr] = {uint_values[1], uint_values[2], uint_values[3]};
      }
    }

    // Skip through unsupported lines
    else {
      for (size_t tmp = 0; tmp < e_amount; tmp++)
        getline(file, line);
    }
  }

  file.close();
}

void Manifold::move_nodes(const std::vector<double> node_movement) {

  for (size_t n = 0; n < current_coordinates.size(); n++) {
    const size_t ni = UniqueNodeNumber()[n];

    current_coordinates[n][0] = initial_coordinates[n][0] + node_movement[3 * ni + 0];
    current_coordinates[n][1] = initial_coordinates[n][1] + node_movement[3 * ni + 1];
    current_coordinates[n][2] = initial_coordinates[n][2] + node_movement[3 * ni + 2];
  }
}

void Manifold::set_nodes(const std::vector<double> node_movement) {

  for (size_t n = 0; n < current_coordinates.size(); n++) {
    const size_t ni = UniqueNodeNumber()[n];

    current_coordinates[n][0] = node_movement[3 * ni + 0];
    current_coordinates[n][1] = node_movement[3 * ni + 1];
    current_coordinates[n][2] = node_movement[3 * ni + 2];
  }
}
