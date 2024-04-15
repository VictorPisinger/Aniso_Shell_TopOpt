#include "MMAseq.h"
#include "exodusWriter.h"
#include "finiteElementAnalysis.h"
#include "manifold.h"
#include "mesh.h"
#include "options.h"
#include "pdeFilter.h"
#include "qualityConstraint.h"
#include "thicknessFilter.h"
#include "volumeConstraint.h"

#include "eigen3/Eigen/Dense"

#include <iostream>

int main(int argc, char* argv[]) {
  Options opts(argc, argv);

  const std::string outputFile  = opts.get<std::string>("output_file", "out.e");
  const std::string inputBase   = opts.get<std::string>("input_files", "mesh_*.ply");
  const double defaultThickness = opts.get("default_thickness", 0.1);

  Mesh mesh;
  std::cout << "Reading shell from " << inputBase << std::endl;
  mesh.read_ply_files(inputBase, defaultThickness);
  mesh.setup_dof_numbering();

  const auto bcDof              = opts.get_boundary_dofs(mesh);
  const auto forces             = opts.get_forces(mesh);
  const auto manifold_pressures = opts.get_pressure_manifolds();

  const double youngs_module  = opts.get("youngs_module", 1.0);
  const double poissons_ratio = opts.get("poissons_ratio", 0.3);

  FiniteElementAnalysis fea(mesh, youngs_module, poissons_ratio, manifold_pressures);
  fea.set_fixed_dof(bcDof);
  fea.set_forces(forces);

  const std::string jobType = opts.get<std::string>("job_type", "analysis");
  if (jobType.compare("analysis") == 0) {
    std::cout << "Performing analysis of structure" << std::endl;

    const auto displacement = fea.solve_system();
    const auto stress       = fea.compute_stress(displacement);
    const auto vm_stress    = fea.compute_vm_stress(displacement);
    const double compliance = fea.compute_compliance(displacement);
    const auto c_split      = fea.compute_compliances(displacement);

    ExodusWriter writer(outputFile, mesh, {"ux", "uy", "uz"},
                        {"thickness", "smxx", "smyy", "smzz", "smxy", "smxz", "smyz", "sbxx", "sbyy", "sbzz", "sbxy",
                         "sbxz", "sbyz", "vm_stress"});
    writer.writeThickness(0);
    writer.writeDisplacement(displacement, 0);
    writer.writeTensor(stress[0], 1);
    writer.writeTensor(stress[1], 7);
    writer.writeElementField(vm_stress, 13);

    std::cout << "Final compliance value: " << compliance << " c_m:" << c_split[0] << " c_b:" << c_split[1]
              << std::endl;

    // mesh.write_ply_files(outputBase);

  } else if (jobType.compare("thickness") == 0) {
    std::cout << "Performing optimization of thickness" << std::endl;

    const auto maxIters = opts.get<int>("max_its", 100);
    const auto volfrac  = opts.get<double>("volume_fraction", 1.0);

    double cscale = 0.0;
    VolumeConstraint volumeConstraint(mesh, volfrac);

    const auto thk_filter_radius = opts.get<double>("thickness_filter_radius", 0.0);
    ThicknessFilter thicknessFilter(mesh, thk_filter_radius);

    size_t numDesignVars = 0;
    for (auto& m : mesh.manifolds)
      numDesignVars += m.Points().size();

    std::vector<double> dc_serial, dg_serial, x_serial;
    dc_serial.reserve(numDesignVars);
    dg_serial.reserve(numDesignVars);
    x_serial.reserve(numDesignVars);

    MMAseq mma(numDesignVars, 1);

    for (const auto& m : mesh.manifolds)
      for (const auto& v : m.Thickness())
        x_serial.emplace_back(v);

    const double max_thickness = opts.get<double>("thk_max", 0.15);
    const double min_thickness = opts.get<double>("thk_min", 0.05);
    const double move_lim      = opts.get<double>("thk_move_limit", 0.01);
    double compliance          = 0.0;

    std::vector<double> xmin(numDesignVars), xmax(numDesignVars);

    std::vector<std::vector<double>> x_intermediate(mesh.manifolds.size());
    for (size_t i = 0; i < mesh.manifolds.size(); i++) {
      x_intermediate[i].resize(mesh.manifolds[i].Thickness().size(), 0.0);
    }

    ExodusWriter writer(
        outputFile, mesh, {"ux", "uy", "uz"},
        {"thickness", "smxx", "smyy", "smzz", "smxy", "smxz", "smyz", "sbxx", "sbyy", "sbzz", "sbxy", "sbxz", "sbyz"});

    for (int iter = 1; iter < maxIters + 1; iter++) {

      // compute solution
      const auto u = fea.solve_system();

      const auto stress = fea.compute_stress(u);
      writer.writeThickness(0);
      writer.writeDisplacement(u, 0);
      writer.writeTensor(stress[0], 1);
      writer.writeTensor(stress[1], 7);
      writer.incrementTimestep();

      // compute gradients
      const double c_unscaled = fea.compute_compliance(u);
      const auto c_split      = fea.compute_compliances(u);
      if (iter == 1) {
        cscale = 10.0 / c_unscaled;
      }

      compliance     = c_unscaled;
      const double c = c_unscaled * cscale;
      double g       = volumeConstraint.compute_value();

      auto dc = thicknessFilter.backward(fea.compute_thickness_gradient(u));
      // auto dc = fea.compute_thickness_gradient(u);

      // scale gradient
      dc_serial.clear();
      for (const auto& m : dc)
        for (const auto& v : m) {
          dc_serial.emplace_back(v * cscale);
        }

      auto dg = thicknessFilter.backward(volumeConstraint.compute_thickness_gradient());
      // auto dg = volumeConstraint.compute_thickness_gradient();

      dg_serial.clear();
      for (const auto& m : dg)
        for (const auto& v : m) {
          dg_serial.emplace_back(v);
        }

      // set move limits to max/min
      for (size_t i = 0; i < numDesignVars; i++) {
        xmin[i] = std::max(x_serial[i] - move_lim, min_thickness);
        xmax[i] = std::min(x_serial[i] + move_lim, max_thickness);
      }

      // update design
      mma.Update(x_serial.data(), dc_serial.data(), &g, dg_serial.data(), xmin.data(), xmax.data());

      {
        int count = 0;
        for (size_t i = 0; i < mesh.manifolds.size(); i++)
          for (auto& v : x_intermediate[i]) {
            v = x_serial[count];

            count++;
          }
      }

      auto filtered_thickness = thicknessFilter.forward(x_intermediate);
      // auto filtered_thickness = x_intermediate;

      for (size_t i = 0; i < mesh.manifolds.size(); i++)
        for (size_t j = 0; j < mesh.manifolds[i].Thickness().size(); j++) {
          mesh.manifolds[i].ThicknessMutable()[j] = x_intermediate[i][j];
        }

      // write output
      std::cout << "Iteration " << iter << " c:" << c << " g:" << g << " c_u:" << c_unscaled << " c_m:" << c_split[0]
                << " c_b:" << c_split[1] << std::endl;
    }
    std::cout << "Final compliance value: " << compliance << std::endl;

  } else if (jobType.compare("shape") == 0) {
    std::cout << "Performing optimization of shape" << std::endl;

    const auto maxIters                = opts.get<int>("max_its", 100);
    const auto volfrac                 = opts.get<double>("volume_fraction", 1.0);
    const auto aspect_ratio_constraint = opts.get<double>("aspect_ratio_constraint", 3.0);

    double cscale = 0.0;
    VolumeConstraint volumeConstraint(mesh, volfrac);
    QualityConstraint qualityConstraint(mesh, aspect_ratio_constraint);

    size_t numDesignVars = 3 * mesh.numberOfUniqueNodes;

    MMAseq mma(numDesignVars, 2);
    mma.SetAsymptotes(0.4, 0.6, 1.03);
    mma.ConstraintModification(true);

    const double max_move = opts.get<double>("move_max", 0.5);
    const double min_move = opts.get<double>("move_min", -0.5);
    const double move_lim = opts.get<double>("move_move_limit", 0.05);

    const int num_filters    = opts.get<double>("num_filters", 1);
    const auto filter_radius = opts.get<double>("filter_radius", 1.0) / (double)num_filters;

    PdeFilter filter(mesh, filter_radius, opts.get_non_design_nodes(mesh));

    std::vector<double> x(numDesignVars, 0.0), xphys(numDesignVars, 0.0), xmin(numDesignVars), xmax(numDesignVars);

    ExodusWriter writer(outputFile, mesh, {"ux", "uy", "uz", "px", "py", "pz", "xx", "xy", "xz"},
                        {"thickness", "vm_stress"});

    double compliance = 0.0;

    for (int iter = 1; iter < maxIters + 1; iter++) {

      // compute solution
      const auto u = fea.solve_system();

      const auto vm_stress = fea.compute_vm_stress(u);
      writer.writeThickness(0);
      writer.writeElementField(vm_stress, 1);
      writer.writeDisplacement(u, 0);
      writer.writeDisplacement(Eigen::VectorXd::Map(&xphys[0], xphys.size()), 3);
      writer.writeDisplacement(Eigen::VectorXd::Map(&x[0], x.size()), 6);
      writer.incrementTimestep();

      // compute gradients
      const double c_unscaled = fea.compute_compliance(u);
      if (iter == 1) {
        cscale = 10.0 / c_unscaled;
      }

      if ((cscale * c_unscaled) < 0.5) {
        cscale *= 10.0;
      }

      const auto c_split = fea.compute_compliances(u);
      compliance         = c_unscaled;
      const double c     = c_unscaled * cscale;
      double v           = volumeConstraint.compute_value();
      double q           = qualityConstraint.compute_AR_value();

      auto dc = fea.compute_shape_gradient(u);
      for (int i = 0; i < num_filters; i++) {
        dc = filter.backward(dc);
      }

      auto dq = qualityConstraint.compute_AR_shape_gradient();
      for (int i = 0; i < num_filters; i++) {
        dq = filter.backward(dq);
      }

      auto dv = volumeConstraint.compute_shape_gradient();
      for (int i = 0; i < num_filters; i++) {
        dv = filter.backward(dv);
      }

      // set move limits to max/min
      for (size_t i = 0; i < numDesignVars; i++) {
        xmin[i] = std::max(x[i] - move_lim, min_move);
        xmax[i] = std::min(x[i] + move_lim, max_move);
      }

      for (auto& v : dc)
        v *= cscale;

      std::vector<double> dconstraint(2 * x.size());
      for (size_t i = 0; i < x.size(); i++) {
        dconstraint[2 * i + 0] = dv[i];
        dconstraint[2 * i + 1] = dq[i];
      }

      std::array<double, 2> constraints = {v, q};

      // update design
      mma.Update(x.data(), dc.data(), constraints.data(), dconstraint.data(), xmin.data(), xmax.data());

      xphys = x;
      for (int i = 0; i < num_filters; i++) {
        xphys = filter.forward(xphys);
      }

      mesh.move_nodes(xphys);

      // write output
      std::cout << "Iteration " << iter << " c:" << c << " v:" << v << " q:" << q << " c_true:" << c_unscaled
                << " c_m:" << c_split[0] << " c_b:" << c_split[1] << std::endl;
    }
    std::cout << "Final compliance value: " << compliance << std::endl;

  } else if (jobType.compare("shape_and_thk") == 0) {
    std::cout << "Performing optimization of shape and thickness" << std::endl;

    const auto maxIters                = opts.get<int>("max_its", 100);
    const auto volfrac                 = opts.get<double>("volume_fraction", 1.0);
    const auto aspect_ratio_constraint = opts.get<double>("aspect_ratio_constraint", 3.0);

    double cscale = 0.0;
    VolumeConstraint volumeConstraint(mesh, volfrac);

    if (opts.exists("allowed_volume")) {
      const auto vol = opts.get<double>("allowed_volume", 0.0);
      volumeConstraint.set_allowed_volume(vol);
    }

    QualityConstraint qualityConstraint(mesh, aspect_ratio_constraint);

    size_t numDesignVars_shape = 3 * mesh.numberOfUniqueNodes;
    size_t numDesignVars_thk   = 0;

    for (auto& m : mesh.manifolds)
      numDesignVars_thk += m.Points().size();

    const size_t numDesignVars = numDesignVars_shape + numDesignVars_thk;

    MMAseq mma(numDesignVars, 2);
    mma.SetAsymptotes(0.4, 0.6, 1.03);
    mma.ConstraintModification(true);

    const double shape_max_move  = opts.get<double>("move_max", 0.5);
    const double shape_min_move  = opts.get<double>("move_min", -0.5);
    const double shape_move_lim  = opts.get<double>("move_move_limit", 0.05);
    const auto filter_radius     = opts.get<double>("filter_radius", 1.0);
    const double max_thickness   = opts.get<double>("thk_max", 0.15);
    const double min_thickness   = opts.get<double>("thk_min", 0.05);
    const double thk_move_lim    = opts.get<double>("thk_move_limit", 0.01);
    const auto thk_filter_radius = opts.get<double>("thickness_filter_radius", 0.0);
    const bool update_thk_filter = opts.get<bool>("update_thk_filter", false);

    std::vector<double> x_shape_max, x_shape_min;
    opts.get_movement_bounds(mesh, shape_min_move, shape_max_move, x_shape_min, x_shape_max);

    PdeFilter filter(mesh, filter_radius, opts.get_non_design_nodes(mesh));
    ThicknessFilter thicknessFilter(mesh, thk_filter_radius);

    std::vector<double> x_serial(numDesignVars, 0.0), xphys_shape(numDesignVars, 0.0), xmin_serial(numDesignVars),
        xmax_serial(numDesignVars, 0.0), dc_serial(numDesignVars), x_shape(numDesignVars, 0.0),
        x_thk(numDesignVars, 0.0), dconstraint_serial(2 * numDesignVars, 0.0);

    for (size_t i = numDesignVars_shape; i < numDesignVars; i++) {
      x_serial[i] = defaultThickness;
    }

    std::vector<std::vector<double>> x_intermediate(mesh.manifolds.size());
    for (size_t i = 0; i < mesh.manifolds.size(); i++) {
      x_intermediate[i].resize(mesh.manifolds[i].Thickness().size(), 0.0);
    }

    ExodusWriter writer(outputFile, mesh, {"ux", "uy", "uz", "px", "py", "pz", "fx", "fy", "fz"},
                        {"thickness", "vm_stress"});

    double c_unscaled = 0.0;

    for (int iter = 1; iter < maxIters + 1; iter++) {

      // compute solution
      const auto u = fea.solve_system();

      const auto vm_stress = fea.compute_vm_stress(u);
      writer.writeThickness(0);
      writer.writeElementField(vm_stress, 1);
      writer.writeDisplacement(u, 0);
      writer.writeDisplacement(Eigen::VectorXd::Map(&xphys_shape[0], xphys_shape.size()), 3);
      writer.writeDisplacement(fea.Forces(), 6);
      writer.incrementTimestep();

      // compute gradients
      const auto c_split = fea.compute_compliances(u);
      c_unscaled         = fea.compute_compliance(u);
      if (iter == 1) {
        cscale = 10.0 / c_unscaled;
      }
      if ((cscale * c_unscaled) < 0.5) {
        cscale *= 10.0;
      }

      const double c = c_unscaled * cscale;
      double v       = volumeConstraint.compute_value();
      double q       = qualityConstraint.compute_AR_value();

      auto dc_shape = filter.backward(fea.compute_shape_gradient(u));
      auto dv_shape = filter.backward(volumeConstraint.compute_shape_gradient());
      auto dq_shape = filter.backward(qualityConstraint.compute_AR_shape_gradient());

      auto dc_thk = thicknessFilter.backward(fea.compute_thickness_gradient(u));
      auto dv_thk = thicknessFilter.backward(volumeConstraint.compute_thickness_gradient());

      // set move limits to max/min
      for (size_t i = 0; i < numDesignVars_shape; i++) {
        xmin_serial[i] = std::max(x_serial[i] - shape_move_lim, x_shape_min[i]);
        xmax_serial[i] = std::min(x_serial[i] + shape_move_lim, x_shape_max[i]);
      }
      for (size_t i = numDesignVars_shape; i < numDesignVars; i++) {
        xmin_serial[i] = std::max(x_serial[i] - thk_move_lim, min_thickness);
        xmax_serial[i] = std::min(x_serial[i] + thk_move_lim, max_thickness);
      }

      for (size_t i = 0; i < numDesignVars_shape; i++) {
        dc_serial[i]                  = cscale * dc_shape[i];
        dconstraint_serial[2 * i + 0] = dv_shape[i];
        dconstraint_serial[2 * i + 1] = dq_shape[i];
      }

      size_t counter = numDesignVars_shape;
      for (const auto& m : dc_thk)
        for (const auto& v : m) {
          dc_serial[counter] = v * cscale;
          counter++;
        }

      counter = numDesignVars_shape;
      for (const auto& m : dv_thk)
        for (const auto& v : m) {
          dconstraint_serial[2 * counter] = v;
          counter++;
        }

      std::array<double, 2> constraints = {v, q};

      // update design
      mma.Update(x_serial.data(), dc_serial.data(), constraints.data(), dconstraint_serial.data(), xmin_serial.data(),
                 xmax_serial.data());

      for (size_t i = 0; i < numDesignVars_shape; i++) {
        x_shape[i] = x_serial[i];
      }

      {
        int count = numDesignVars_shape;
        for (size_t i = 0; i < mesh.manifolds.size(); i++)
          for (auto& v : x_intermediate[i]) {
            v = x_serial[count];

            count++;
          }
      }

      auto filtered_thickness = thicknessFilter.forward(x_intermediate);

      for (size_t i = 0; i < mesh.manifolds.size(); i++)
        for (size_t j = 0; j < mesh.manifolds[i].Thickness().size(); j++) {

          const double corrected_thickness = std::max(min_thickness, std::min(max_thickness, filtered_thickness[i][j]));
          mesh.manifolds[i].ThicknessMutable()[j] = corrected_thickness;
        }

      xphys_shape = filter.forward(x_shape);
      mesh.move_nodes(xphys_shape);

      if (update_thk_filter) {
        thicknessFilter.update_filter_system();
      }

      // write output
      std::cout << "Iteration " << iter << " c:" << c << " v:" << v << " q:" << q << " c_true:" << c_unscaled
                << " c_m:" << c_split[0] << " c_b:" << c_split[1] << std::endl;
    }

    std::cout << "Final compliance value: " << c_unscaled << std::endl;
    std::cout << "Final volume: " << volumeConstraint.compute_volume() << std::endl;
  }

  else {
    std::cout << "Job of type <" << jobType << " is not implemented" << std::endl;
  }

  return 0;
}