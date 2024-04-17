#include "options.h"

#include <algorithm>

#include "shellElement.h"

#include "eigen3/Eigen/Dense"

#include <typeinfo>

Options::Options(int argc, char* argv[]) {

  std::string configFile = "../jobs/cylinder.yaml";
    
  char** start     = argv;
  char** end       = argv + argc;
  std::string flag = "-c";
  char** itr       = std::find(start, end, flag);
  if (itr != end && ++itr != end) {
    configFile = std::string(*itr);
  }

  try {
    rootnode = YAML::LoadFile(configFile);
    options  = rootnode["shell-opt"];
  } catch (std::exception e) {
    // if file is bad, just dont load it.
  }
}


// Retrive material properties and store as matrix
std::vector<std::array<double,10>> Options::get_material_prop() {
  std::vector<std::array<double,10>> matProp;
  std::array<double,10> material ;
  int nMat = std::size(rootnode["shell-opt"]["Material"]);
  for (int i = 0; i<nMat; i++){
    std::cout << "Material = " << rootnode["shell-opt"]["Material"][i]["mat"] << std::endl;
    material = rootnode["shell-opt"]["Material"][i]["mat"].as<std::array<double,10>>();
    matProp.push_back(material);
  }
  return matProp;
}

// Retrive element alignment vector
std::array<Eigen::Vector<double, 3>, 3>  Options::get_elemAlignment() {
  std::array<Eigen::Vector<double, 3>, 3> localEvec;
  for (int i=0;i<3;i++){
  std::array<double,3> axis = rootnode["shell-opt"]["elemAlign"][i].as<std::array<double,3>>();
  localEvec[i][0] = axis[0];
  localEvec[i][1] = axis[1];
  localEvec[i][2] = axis[2];
  }
  return localEvec;
}


// Retrive element layup
std::vector<std::array<std::vector<double>, 3>>  Options::get_layUp() {
  std::vector<std::array<std::vector<double>, 3>> layups;
  std::array<std::vector<double>, 3> layer;
  int nLayups = std::size(rootnode["shell-opt"]["Layup"]);
  for (int i=0;i<nLayups;i++){
    const std::vector<double> theta = rootnode["shell-opt"]["Layup"][i]["theta"].as<std::vector<double>>();
    const std::vector<double> thk = rootnode["shell-opt"]["Layup"][i]["thk"].as<std::vector<double>>();
    const std::vector<double> mat= rootnode["shell-opt"]["Layup"][i]["mat"].as<std::vector<double>>();
    layer[0] = theta;
    layer[1] = thk;
    layer[2] = mat;
    layups.push_back(layer);
  }
  return layups;
}

std::vector<uint32_t> Options::get_boundary_dofs(const Mesh& mesh) const {
  std::vector<uint32_t> bcDof;

  for (const auto& bc : rootnode["shell-opt"]["domain"]["boundary_conditions"]) {
    const std::string type = bc["type"].as<std::string>();

    if (type.compare("box") == 0) {
      const double x_max               = bc["x_max"].as<double>();
      const double x_min               = bc["x_min"].as<double>();
      const double y_max               = bc["y_max"].as<double>();
      const double y_min               = bc["y_min"].as<double>();
      const double z_max               = bc["z_max"].as<double>();
      const double z_min               = bc["z_min"].as<double>();
      const std::vector<uint32_t> dofs = bc["dofs"].as<std::vector<uint32_t>>();
      
      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Points().size(); i++) {
          const auto& pt = m.Points()[i];

          if (isInBox(pt)) {
            for (const auto& dof : dofs) {
              bcDof.push_back(3 * m.UniqueNodeNumber()[i] + dof);
            }
          }
        }
      }

    } else if (type.compare("box_zero_rotation") == 0) {
      const double x_max = bc["x_max"].as<double>();
      const double x_min = bc["x_min"].as<double>();
      const double y_max = bc["y_max"].as<double>();
      const double y_min = bc["y_min"].as<double>();
      const double z_max = bc["z_max"].as<double>();
      const double z_min = bc["z_min"].as<double>();

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Faces().size(); i++) {
          const auto& f         = m.Faces()[i];
          const bool pt1_in_box = isInBox(m.Points()[f[0]]);
          const bool pt2_in_box = isInBox(m.Points()[f[1]]);
          const bool pt3_in_box = isInBox(m.Points()[f[2]]);

          if (pt1_in_box && pt2_in_box)
            bcDof.push_back(m.EdgeDofNumber()[i][0]);

          if (pt2_in_box && pt3_in_box)
            bcDof.push_back(m.EdgeDofNumber()[i][1]);

          if (pt1_in_box && pt3_in_box)
            bcDof.push_back(m.EdgeDofNumber()[i][2]);
        }
      }

    } else if (type.compare("outside_circle") == 0) {
      const double radius              = bc["radius"].as<double>();
      const std::vector<double> center = bc["center"].as<std::vector<double>>();
      const std::vector<uint32_t> dofs = bc["dofs"].as<std::vector<uint32_t>>();

      Eigen::Vector3d center_v = {center[0], center[1], center[2]};

      auto isOutOfCircle = [&](const std::array<double, 3> pt) {
        Eigen::Vector3d pt_v = {pt[0], pt[1], pt[2]};
        const double dist    = (pt_v - center_v).norm();
        return dist >= radius;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Points().size(); i++) {
          const auto& pt = m.Points()[i];

          if (isOutOfCircle(pt)) {
            for (const auto& dof : dofs) {
              bcDof.push_back(3 * m.UniqueNodeNumber()[i] + dof);
            }
          }
        }
      }

    } else if (type.compare("outside_circle_zero_rotation") == 0) {
      const double radius              = bc["radius"].as<double>();
      const std::vector<double> center = bc["center"].as<std::vector<double>>();
      const std::vector<uint32_t> dofs = bc["dofs"].as<std::vector<uint32_t>>();

      Eigen::Vector3d center_v = {center[0], center[1], center[2]};

      auto isOutOfCircle = [&](const std::array<double, 3> pt) {
        Eigen::Vector3d pt_v = {pt[0], pt[1], pt[2]};
        const double dist    = (pt_v - center_v).norm();
        return dist >= radius;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Faces().size(); i++) {
          const auto& f         = m.Faces()[i];
          const bool pt1_in_box = isOutOfCircle(m.Points()[f[0]]);
          const bool pt2_in_box = isOutOfCircle(m.Points()[f[1]]);
          const bool pt3_in_box = isOutOfCircle(m.Points()[f[2]]);

          if (pt1_in_box && pt2_in_box)
            bcDof.push_back(m.EdgeDofNumber()[i][0]);

          if (pt2_in_box && pt3_in_box)
            bcDof.push_back(m.EdgeDofNumber()[i][1]);

          if (pt1_in_box && pt3_in_box)
            bcDof.push_back(m.EdgeDofNumber()[i][2]);
        }
      }
    }

    else {
      std::cout << "Warning! did not recognize boudnary_condition type \"" << type << "\"" << std::endl;
    }
  }

  return bcDof;
}

std::vector<std::pair<uint32_t, double>> Options::get_forces(const Mesh& mesh) const {
  std::vector<std::pair<uint32_t, double>> forces;

  for (const auto& bc : rootnode["shell-opt"]["domain"]["forces"]) {
    const std::string type = bc["type"].as<std::string>();

    if (type.compare("box") == 0) {

      const double x_max               = bc["x_max"].as<double>();
      const double x_min               = bc["x_min"].as<double>();
      const double y_max               = bc["y_max"].as<double>();
      const double y_min               = bc["y_min"].as<double>();
      const double z_max               = bc["z_max"].as<double>();
      const double z_min               = bc["z_min"].as<double>();
      const std::vector<uint32_t> dofs = bc["dofs"].as<std::vector<uint32_t>>();
      const std::vector<double> values = bc["values"].as<std::vector<double>>();

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Points().size(); i++) {
          const auto& pt = m.Points()[i];

          if (isInBox(pt)) {
            for (size_t j = 0; j < dofs.size(); j++) {
              const uint32_t index = 3 * m.UniqueNodeNumber()[i] + dofs[j];
              forces.emplace_back(index, values[j]);
            }
          }
        }
      }
    }

    else if (type.compare("box_edge_traction") == 0) {

      const double x_max               = bc["x_max"].as<double>();
      const double x_min               = bc["x_min"].as<double>();
      const double y_max               = bc["y_max"].as<double>();
      const double y_min               = bc["y_min"].as<double>();
      const double z_max               = bc["z_max"].as<double>();
      const double z_min               = bc["z_min"].as<double>();
      const std::vector<double> values = bc["traction"].as<std::vector<double>>();

      const Eigen::Vector3d traction = {values[0], values[1], values[2]};

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Faces().size(); i++) {
          const auto& f = m.Faces()[i];

          const bool pt1_in_box = isInBox(m.Points()[f[0]]);
          const bool pt2_in_box = isInBox(m.Points()[f[1]]);
          const bool pt3_in_box = isInBox(m.Points()[f[2]]);

          if (pt1_in_box && pt2_in_box) {
            const Eigen::Vector3d p1 = {m.Points()[f[0]][0], m.Points()[f[0]][1], m.Points()[f[0]][2]};
            const Eigen::Vector3d p2 = {m.Points()[f[1]][0], m.Points()[f[1]][1], m.Points()[f[1]][2]};

            auto force = Shell::compute_edge_traction<double>({p1, p2}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[0]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[1]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
            }
          }

          if (pt2_in_box && pt3_in_box) {
            const Eigen::Vector3d p2 = {m.Points()[f[1]][0], m.Points()[f[1]][1], m.Points()[f[1]][2]};
            const Eigen::Vector3d p3 = {m.Points()[f[2]][0], m.Points()[f[2]][1], m.Points()[f[2]][2]};

            auto force = Shell::compute_edge_traction<double>({p2, p3}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[1]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[2]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
            }
          }

          if (pt1_in_box && pt3_in_box) {
            const Eigen::Vector3d p1 = {m.Points()[f[0]][0], m.Points()[f[0]][1], m.Points()[f[0]][2]};
            const Eigen::Vector3d p3 = {m.Points()[f[2]][0], m.Points()[f[2]][1], m.Points()[f[2]][2]};

            auto force = Shell::compute_edge_traction<double>({p1, p3}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[0]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[2]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
            }
          }
        }
      }
    }

    else if (type.compare("manifold_pressure") == 0) {
      // do not add force here.
    }

    else if (type.compare("box_surface_traction") == 0) {

      const double x_max               = bc["x_max"].as<double>();
      const double x_min               = bc["x_min"].as<double>();
      const double y_max               = bc["y_max"].as<double>();
      const double y_min               = bc["y_min"].as<double>();
      const double z_max               = bc["z_max"].as<double>();
      const double z_min               = bc["z_min"].as<double>();
      const std::vector<double> values = bc["traction"].as<std::vector<double>>();

      const Eigen::Vector3d traction = {values[0], values[1], values[2]};

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Faces().size(); i++) {
          const auto& f = m.Faces()[i];

          const bool pt1_in_box = isInBox(m.Points()[f[0]]);
          const bool pt2_in_box = isInBox(m.Points()[f[1]]);
          const bool pt3_in_box = isInBox(m.Points()[f[2]]);

          if (pt1_in_box && pt2_in_box && pt3_in_box) {
            const Eigen::Vector3d p1 = {m.Points()[f[0]][0], m.Points()[f[0]][1], m.Points()[f[0]][2]};
            const Eigen::Vector3d p2 = {m.Points()[f[1]][0], m.Points()[f[1]][1], m.Points()[f[1]][2]};
            const Eigen::Vector3d p3 = {m.Points()[f[2]][0], m.Points()[f[2]][1], m.Points()[f[2]][2]};

            auto force = Shell::compute_element_traction<double>({p1, p2, p3}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[0]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[1]] + j;
              const uint32_t i3 = 3 * m.UniqueNodeNumber()[f[2]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
              forces.emplace_back(i3, force(j));
            }
          }
        }
      }
    }

    else if (type.compare("box_surface_rotation") == 0) {

      const double x_max                 = bc["x_max"].as<double>();
      const double x_min                 = bc["x_min"].as<double>();
      const double y_max                 = bc["y_max"].as<double>();
      const double y_min                 = bc["y_min"].as<double>();
      const double z_max                 = bc["z_max"].as<double>();
      const double z_min                 = bc["z_min"].as<double>();
      const double magnitude             = bc["magnitude"].as<double>();
      const std::vector<double> center_v = bc["center"].as<std::vector<double>>();
      const std::vector<double> normal_v = bc["normal"].as<std::vector<double>>();

      const Eigen::Vector3d center = {center_v[0], center_v[1], center_v[2]};
      const Eigen::Vector3d normal = {normal_v[0], normal_v[1], normal_v[2]};

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Faces().size(); i++) {
          const auto& f = m.Faces()[i];

          const bool pt1_in_box = isInBox(m.Points()[f[0]]);
          const bool pt2_in_box = isInBox(m.Points()[f[1]]);
          const bool pt3_in_box = isInBox(m.Points()[f[2]]);

          if (pt1_in_box && pt2_in_box && pt3_in_box) {
            const Eigen::Vector3d p1 = {m.Points()[f[0]][0], m.Points()[f[0]][1], m.Points()[f[0]][2]};
            const Eigen::Vector3d p2 = {m.Points()[f[1]][0], m.Points()[f[1]][1], m.Points()[f[1]][2]};
            const Eigen::Vector3d p3 = {m.Points()[f[2]][0], m.Points()[f[2]][1], m.Points()[f[2]][2]};

            const auto el_center = (1.0 / 3.0) * (p1 + p2 + p3);

            const auto director = (el_center - center);
            const auto distance = director.norm();
            const auto traction = distance * director.cross(normal).normalized();

            auto force = Shell::compute_element_traction<double>({p1, p2, p3}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[0]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[1]] + j;
              const uint32_t i3 = 3 * m.UniqueNodeNumber()[f[2]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
              forces.emplace_back(i3, force(j));
            }
          }
        }
      }
    }

    else if (type.compare("box_edge_rotation") == 0) {

      const double x_max                 = bc["x_max"].as<double>();
      const double x_min                 = bc["x_min"].as<double>();
      const double y_max                 = bc["y_max"].as<double>();
      const double y_min                 = bc["y_min"].as<double>();
      const double z_max                 = bc["z_max"].as<double>();
      const double z_min                 = bc["z_min"].as<double>();
      const double magnitude             = bc["magnitude"].as<double>();
      const std::vector<double> center_v = bc["center"].as<std::vector<double>>();
      const std::vector<double> normal_v = bc["normal"].as<std::vector<double>>();

      const Eigen::Vector3d center = {center_v[0], center_v[1], center_v[2]};
      const Eigen::Vector3d normal = {normal_v[0], normal_v[1], normal_v[2]};

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Faces().size(); i++) {
          const auto& f = m.Faces()[i];

          const bool pt1_in_box = isInBox(m.Points()[f[0]]);
          const bool pt2_in_box = isInBox(m.Points()[f[1]]);
          const bool pt3_in_box = isInBox(m.Points()[f[2]]);

          if (pt1_in_box && pt2_in_box) {
            const Eigen::Vector3d p1 = {m.Points()[f[0]][0], m.Points()[f[0]][1], m.Points()[f[0]][2]};
            const Eigen::Vector3d p2 = {m.Points()[f[1]][0], m.Points()[f[1]][1], m.Points()[f[1]][2]};

            const auto edge_center         = 0.5 * (p1 + p2);
            const auto director            = (edge_center - center);
            const auto distance            = director.norm();
            const Eigen::Vector3d traction = distance * director.cross(normal).normalized();

            auto force = Shell::compute_edge_traction<double>({p1, p2}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[0]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[1]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
            }
          }

          if (pt2_in_box && pt3_in_box) {
            const Eigen::Vector3d p2 = {m.Points()[f[1]][0], m.Points()[f[1]][1], m.Points()[f[1]][2]};
            const Eigen::Vector3d p3 = {m.Points()[f[2]][0], m.Points()[f[2]][1], m.Points()[f[2]][2]};

            const auto edge_center = 0.5 * (p2 + p3);
            const auto director    = (edge_center - center);
            const auto distance    = director.norm();
            const auto traction    = distance * director.cross(normal).normalized();

            auto force = Shell::compute_edge_traction<double>({p2, p3}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[1]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[2]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
            }
          }

          if (pt1_in_box && pt3_in_box) {
            const Eigen::Vector3d p1 = {m.Points()[f[0]][0], m.Points()[f[0]][1], m.Points()[f[0]][2]};
            const Eigen::Vector3d p3 = {m.Points()[f[2]][0], m.Points()[f[2]][1], m.Points()[f[2]][2]};

            const auto edge_center = 0.5 * (p1 + p3);
            const auto director    = (edge_center - center);
            const auto distance    = director.norm();
            const auto traction    = distance * director.cross(normal).normalized();

            auto force = Shell::compute_edge_traction<double>({p1, p3}, traction);
            for (size_t j = 0; j < 3; j++) {
              const uint32_t i1 = 3 * m.UniqueNodeNumber()[f[0]] + j;
              const uint32_t i2 = 3 * m.UniqueNodeNumber()[f[2]] + j;
              forces.emplace_back(i1, force(j));
              forces.emplace_back(i2, force(j));
            }
          }
        }
      }
    }

    else {
      std::cout << "Warning! did not recognize force type \"" << type << "\"" << std::endl;
    }
  }

  return forces;
}

std::vector<std::pair<size_t, double>> Options::get_pressure_manifolds() const {

  std::vector<std::pair<size_t, double>> manifolds;

  for (const auto& bc : rootnode["shell-opt"]["domain"]["forces"]) {
    const std::string type = bc["type"].as<std::string>();

    if (type.compare("manifold_pressure") == 0) {
      const int manifold_index = bc["manifold_index"].as<int>();
      const double pressure    = bc["pressure"].as<double>();

      manifolds.emplace_back(manifold_index, pressure);
    }
  }

  return manifolds;
}


std::array<std::shared_ptr<std::vector<uint32_t>>, 3> Options::get_non_design_nodes(const Mesh& mesh) const {

  std::array<std::vector<uint32_t>, 3> nodes_xyz;

  for (const auto& bc : rootnode["shell-opt"]["domain"]["non_design_nodes"]) {
    const std::string type = bc["type"].as<std::string>();

    if (type.compare("box") == 0) {

      const double x_max               = bc["x_max"].as<double>();
      const double x_min               = bc["x_min"].as<double>();
      const double y_max               = bc["y_max"].as<double>();
      const double y_min               = bc["y_min"].as<double>();
      const double z_max               = bc["z_max"].as<double>();
      const double z_min               = bc["z_min"].as<double>();
      const std::vector<uint32_t> dofs = bc["dofs"].as<std::vector<uint32_t>>();

      auto isInBox = [&](const std::array<double, 3> pt) {
        const bool in_x = pt[0] <= x_max && pt[0] >= x_min;
        const bool in_y = pt[1] <= y_max && pt[1] >= y_min;
        const bool in_z = pt[2] <= z_max && pt[2] >= z_min;
        return in_x && in_y && in_z;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Points().size(); i++) {
          if (isInBox(m.Points()[i])) {
            for (const auto d : dofs) {
              nodes_xyz[d].push_back(m.UniqueNodeNumber()[i]);
            }
          }
        }
      }

    } else if (type.compare("outside_circle") == 0) {
      const double radius              = bc["radius"].as<double>();
      const std::vector<double> center = bc["center"].as<std::vector<double>>();
      const std::vector<uint32_t> dofs = bc["dofs"].as<std::vector<uint32_t>>();

      Eigen::Vector3d center_v = {center[0], center[1], center[2]};

      auto isOutOfCircle = [&](const std::array<double, 3> pt) {
        Eigen::Vector3d pt_v = {pt[0], pt[1], pt[2]};
        const double dist    = (pt_v - center_v).norm();
        return dist >= radius;
      };

      for (const auto& m : mesh.manifolds) {
        for (size_t i = 0; i < m.Points().size(); i++) {
          const auto& pt = m.Points()[i];

          if (isOutOfCircle(pt)) {
            for (const auto d : dofs) {
              nodes_xyz[d].push_back(m.UniqueNodeNumber()[i]);
            }
          }
        }
      }

    } else {
      std::cout << "Warning! did not recognize non-design node type \"" << type << "\"" << std::endl;
    }
  }

  std::array<std::shared_ptr<std::vector<uint32_t>>, 3> output;
  output[0] = std::make_shared<std::vector<uint32_t>>(nodes_xyz[0]);

  // if y==x, copy the reference only
  if (nodes_xyz[0].size() == nodes_xyz[1].size() &&
      std::equal(nodes_xyz[1].begin(), nodes_xyz[1].end(), nodes_xyz[0].begin())) {
    output[1] = output[0];
  } else {
    output[1] = std::make_shared<std::vector<uint32_t>>(nodes_xyz[1]);
  }

  // if z==x or z==y, copy the reference only
  if (nodes_xyz[0].size() == nodes_xyz[2].size() &&
      std::equal(nodes_xyz[2].begin(), nodes_xyz[2].end(), nodes_xyz[0].begin())) {
    output[2] = output[0];
  } else if (nodes_xyz[1].size() == nodes_xyz[2].size() &&
             std::equal(nodes_xyz[2].begin(), nodes_xyz[2].end(), nodes_xyz[1].begin())) {
    output[2] = output[1];
  } else {
    output[2] = std::make_shared<std::vector<uint32_t>>(nodes_xyz[2]);
  }

  return output;
}

void Options::get_movement_bounds(const Mesh& mesh, const double min_val, const double max_val,
                                  std::vector<double>& move_min, std::vector<double>& move_max) const {
  const size_t numDesignVars = 3 * mesh.numberOfUniqueNodes;
  move_min.resize(numDesignVars);
  move_max.resize(numDesignVars);

  for (size_t i = 0; i < numDesignVars; i++) {
    move_min[i] = min_val;
    move_max[i] = max_val;
  }

  if (rootnode["shell-opt"]["domain"]["bounding_box"]) {
    auto box           = rootnode["shell-opt"]["domain"]["bounding_box"];
    const double x_max = box["x_max"].as<double>();
    const double x_min = box["x_min"].as<double>();
    const double y_max = box["y_max"].as<double>();
    const double y_min = box["y_min"].as<double>();
    const double z_max = box["z_max"].as<double>();
    const double z_min = box["z_min"].as<double>();

    for (const auto& m : mesh.manifolds) {
      for (size_t i = 0; i < m.InitialPoints().size(); i++) {
        const auto& pt = m.InitialPoints()[i];
        const auto ni  = m.UniqueNodeNumber()[i];

        move_min[3 * ni + 0] = std::max(min_val, x_min - pt[0]);
        move_max[3 * ni + 0] = std::min(max_val, x_max - pt[0]);
        move_min[3 * ni + 1] = std::max(min_val, y_min - pt[1]);
        move_max[3 * ni + 1] = std::min(max_val, y_max - pt[1]);
        move_min[3 * ni + 2] = std::max(min_val, z_min - pt[2]);
        move_max[3 * ni + 2] = std::min(max_val, z_max - pt[2]);
      }
    }
  }
}