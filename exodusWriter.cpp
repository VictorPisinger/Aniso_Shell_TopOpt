#include "exodusWriter.h"

#include "exodusII.h"

ExodusWriter::ExodusWriter(const std::string fname, const Mesh& mesh, std::vector<std::string> pointFieldNames,
                           std::vector<std::string> elementFieldNames)
    : filename(fname), mesh_(mesh) {

  static constexpr char modName[] = "initialMeshWrite";

  int programFloatSize = sizeof(double);
  int fileFloatSize    = sizeof(float);

  // create exodus file, overwrite existing file.
  int exodusID = ex_create(filename.c_str(), EX_CLOBBER, &programFloatSize, &fileFloatSize);

  char titleInFile[] = "Shopt output";

  int exoError = ex_put_init(exodusID, titleInFile, 3,   // dimensions
                             mesh.numberOfUniqueNodes,   // number of nodes
                             mesh.number_of_triangles(), // number of elements
                             mesh.manifolds.size(), 0, 0);
  ex_err(modName, modName, exoError);

  { // write points to file
    std::vector<double> coordinatesX(mesh.numberOfUniqueNodes);
    std::vector<double> coordinatesY(mesh.numberOfUniqueNodes);
    std::vector<double> coordinatesZ(mesh.numberOfUniqueNodes);
    for (const auto& m : mesh.manifolds)
      for (size_t i = 0; i < m.Points().size(); i++) {
        const auto& idx   = m.UniqueNodeNumber()[i];
        coordinatesX[idx] = m.Points()[i][0];
        coordinatesY[idx] = m.Points()[i][1];
        coordinatesZ[idx] = m.Points()[i][2];
      }

    exoError = ex_put_coord(exodusID, coordinatesX.data(), coordinatesY.data(), coordinatesZ.data());
    ex_err(modName, modName, exoError);
  }

  // write faces to file
  for (size_t mi = 0; mi < mesh.manifolds.size(); mi++) {
    const auto& m = mesh.manifolds[mi];

    exoError =
        ex_put_block(exodusID, EX_ELEM_BLOCK, mi /* block id*/, "TRIANGLE", m.Faces().size(), 3 /* nodes pr element */,
                     0 /* edges pr element */, 0 /* faces pr element */, 0 /* attributes pr element */);
    ex_err(modName, modName, exoError);

    std::vector<int32_t> connectivity;
    connectivity.reserve(3 * m.Faces().size());
    for (const auto& f : m.Faces()) {
      for (int j = 0; j < 3; j++) {
        const uint32_t nodeIndex = m.UniqueNodeNumber()[f[j]];
        connectivity.push_back(nodeIndex + 1); // add 1, as exodus files are 1-indexed.
      }
    }
    exoError = ex_put_conn(exodusID, EX_ELEM_BLOCK, mi /* block id*/, connectivity.data(), NULL, NULL);
    ex_err(modName, modName, exoError);
  }

  // allocate the fields
  exoError =
      ex_put_all_var_param(exodusID, 0, pointFieldNames.size(), elementFieldNames.size(), NULL, 0, NULL, 0, NULL);
  ex_err(modName, modName, exoError);

  for (size_t i = 0; i < pointFieldNames.size(); i++) {
    exoError = ex_put_variable_name(exodusID, EX_NODAL, i + 1, pointFieldNames[i].data());
    ex_err(modName, modName, exoError);
    exoError = ex_put_variable_param(exodusID, EX_NODAL, mesh.numberOfUniqueNodes);
    ex_err(modName, modName, exoError);
  }

  for (size_t i = 0; i < elementFieldNames.size(); i++) {
    exoError = ex_put_variable_name(exodusID, EX_ELEM_BLOCK, i + 1, elementFieldNames[i].data());
    ex_err(modName, modName, exoError);
    // exoError = ex_put_variable_param(exodusID, EX_ELEM_BLOCK, mesh.manifolds[i].Faces().size());
    // ex_err(modName, modName, exoError);
  }

  ex_close(exodusID);
}

void ExodusWriter::writeNodeField(const std::vector<std::vector<double>>& value, int fieldNo) const {
  std::vector<double> valuesAsArray(mesh_.numberOfUniqueNodes);

  for (size_t m = 0; m < mesh_.manifolds.size(); m++) {
    const auto& manifold = mesh_.manifolds[m];
    for (size_t i = 0; i < manifold.Points().size(); i++) {
      valuesAsArray[manifold.UniqueNodeNumber()[i]] = value[m][i];
    }
  }

  float version;
  int programFloatSize            = sizeof(double);
  int fileFloatSize               = sizeof(float);
  static constexpr char modName[] = "writeDisplacement";
  int exodusID                    = ex_open(filename.c_str(), EX_WRITE, &programFloatSize, &fileFloatSize, &version);
  int exoError;

  exoError =
      ex_put_var(exodusID, this->timestep, EX_NODAL, fieldNo + 1, 0, mesh_.numberOfUniqueNodes, valuesAsArray.data());
  ex_err(modName, modName, exoError);

  ex_close(exodusID);
}

void ExodusWriter::writeThickness(int fieldNo) const {
  std::vector<double> valuesAsArray;

  float version;
  int programFloatSize            = sizeof(double);
  int fileFloatSize               = sizeof(float);
  static constexpr char modName[] = "writeDisplacement";
  int exodusID                    = ex_open(filename.c_str(), EX_WRITE, &programFloatSize, &fileFloatSize, &version);
  int exoError;

  double timevalue = timestep;
  ex_put_time(exodusID, timestep, &timevalue);

  for (size_t mi = 0; mi < mesh_.manifolds.size(); mi++) {
    const auto& m             = mesh_.manifolds[mi];
    const size_t num_elements = m.Faces().size();

    valuesAsArray.resize(num_elements);

    for (size_t i = 0; i < m.Faces().size(); i++) {
      const auto& f    = m.Faces()[i];
      valuesAsArray[i] = (m.Thickness()[f[0]] + m.Thickness()[f[1]] + m.Thickness()[f[2]]) / 3.0;
    }

    double timevalue = timestep;
    ex_put_time(exodusID, timestep, &timevalue);
    for (int j = 0; j < 6; j++) {
      exoError =
          ex_put_var(exodusID, this->timestep, EX_ELEM_BLOCK, fieldNo + 1, mi, num_elements, valuesAsArray.data());
      ex_err(modName, modName, exoError);
    }
  }
  ex_close(exodusID);
}

void ExodusWriter::writeDisplacement(const Eigen::VectorXd& displacement, int startField) const {

  std::array<std::vector<double>, 3> nodalDisplacemnts;

  for (auto& disp : nodalDisplacemnts) {
    disp.resize(mesh_.numberOfUniqueNodes);
  }

  for (size_t i = 0; i < mesh_.numberOfUniqueNodes; i++) {
    nodalDisplacemnts[0][i] = displacement(3 * i + 0);
    nodalDisplacemnts[1][i] = displacement(3 * i + 1);
    nodalDisplacemnts[2][i] = displacement(3 * i + 2);
  }

  float version;
  int programFloatSize            = sizeof(double);
  int fileFloatSize               = sizeof(float);
  static constexpr char modName[] = "writeDisplacement";
  int exodusID                    = ex_open(filename.c_str(), EX_WRITE, &programFloatSize, &fileFloatSize, &version);
  int exoError;

  double timevalue = timestep;
  ex_put_time(exodusID, timestep, &timevalue);
  exoError = ex_put_var(exodusID, this->timestep, EX_NODAL, startField + 1, 0, mesh_.numberOfUniqueNodes,
                        nodalDisplacemnts[0].data());
  ex_err(modName, modName, exoError);
  exoError = ex_put_var(exodusID, this->timestep, EX_NODAL, startField + 2, 0, mesh_.numberOfUniqueNodes,
                        nodalDisplacemnts[1].data());
  ex_err(modName, modName, exoError);
  exoError = ex_put_var(exodusID, this->timestep, EX_NODAL, startField + 3, 0, mesh_.numberOfUniqueNodes,
                        nodalDisplacemnts[2].data());
  ex_err(modName, modName, exoError);

  ex_close(exodusID);
}

void ExodusWriter::writeTensor(const std::vector<std::vector<std::array<double, 6>>> tensor, int startField) const {

  std::array<std::vector<double>, 6> tensorComponents;

  float version;
  int programFloatSize            = sizeof(double);
  int fileFloatSize               = sizeof(float);
  static constexpr char modName[] = "writeDisplacement";
  int exodusID                    = ex_open(filename.c_str(), EX_WRITE, &programFloatSize, &fileFloatSize, &version);
  int exoError;

  for (size_t mi = 0; mi < mesh_.manifolds.size(); mi++) {
    const auto& m             = mesh_.manifolds[mi];
    const size_t num_elements = m.Faces().size();

    for (auto& t : tensorComponents) {
      t.resize(num_elements);
    }

    for (size_t i = 0; i < num_elements; i++) {
      for (int j = 0; j < 4; j++) {
        tensorComponents[j][i] = tensor[mi][i][j];
      }
    }
    for (size_t i = 0; i < num_elements; i++) {
      tensorComponents[5][i] = tensor[mi][i][4];
      tensorComponents[4][i] = tensor[mi][i][5];
    }

    double timevalue = timestep;
    ex_put_time(exodusID, timestep, &timevalue);
    for (int j = 0; j < 6; j++) {
      exoError = ex_put_var(exodusID, this->timestep, EX_ELEM_BLOCK, startField + 1 + j, mi, num_elements,
                            tensorComponents[j].data());
      ex_err(modName, modName, exoError);
    }
  }
  ex_close(exodusID);
}

void ExodusWriter::writeElementField(const std::vector<std::vector<double>> field, int field_number) const {

  float version;
  int programFloatSize            = sizeof(double);
  int fileFloatSize               = sizeof(float);
  static constexpr char modName[] = "writeDisplacement";
  int exodusID                    = ex_open(filename.c_str(), EX_WRITE, &programFloatSize, &fileFloatSize, &version);
  int exoError;

  for (size_t mi = 0; mi < mesh_.manifolds.size(); mi++) {
    const auto& m             = mesh_.manifolds[mi];
    const size_t num_elements = m.Faces().size();

    double timevalue = timestep;
    ex_put_time(exodusID, timestep, &timevalue);
    exoError =
        ex_put_var(exodusID, this->timestep, EX_ELEM_BLOCK, field_number + 1, mi, num_elements, field[mi].data());
    ex_err(modName, modName, exoError);
  }
  ex_close(exodusID);
}