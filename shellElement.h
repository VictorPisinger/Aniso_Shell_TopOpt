#pragma once

#include <array>

#include "eigen3/Eigen/Dense"

/*
This is the implementation of the shell element, in a series of functions.

The floating point type is templated, to allow the drop in replacement of double for automatic differentiation.
*/

namespace Shell {

/**
 * @brief Compute the local cartesian coordinate system from the triangle
 * coordinates.
 *
 * @param globalCoords The three corner nodes of the triangle in global
 * coordinates.
 * @return std::array<Eigen::Vector<T,3>, 3> {e1,e2,e3}, e1 and e2 are in-plane,
 * e3 is out-of-plane
 */
template <typename T>
std::array<Eigen::Vector<T, 3>, 3> compute_local_basis(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords) {

  std::array<Eigen::Vector<T, 3>, 3> e; // {e1,e2,e3}

  // e1 is the vector from node 0 to node 1
  e[0] = globalCoords[1] - globalCoords[0];
  e[0].normalize();

  // set temp vector from node 0 to node 2
  e[1] = globalCoords[2] - globalCoords[0];
  e[1].normalize();

  // compute e3 as out-of-plane vector
  e[2] = e[0].cross(e[1]);
  e[2].normalize();

  // compute e2 to make the system orthogonal.
  e[1] = e[2].cross(e[0]);
  e[1].normalize();

  return e;
}

template <typename T>
T compute_area(const std::array<Eigen::Vector<T, 3>, 3>& coords) {

  const auto v1 = coords[0] - coords[1];
  const auto v2 = coords[0] - coords[2];

  return 0.5 * v1.cross(v2).norm();
}

template <typename T>
T compute_aspect_ratio(const std::array<Eigen::Vector<T, 3>, 3>& coords) {

  const T a = (coords[0] - coords[1]).norm();
  const T b = (coords[0] - coords[2]).norm();
  const T c = (coords[1] - coords[2]).norm();

  // a,b,c = side lengths
  // AR = abc/((b+c-a)(c+a-b)(a+b-c))
  return (a * b * c) / ((b + c - a) * (c + a - b) * (a + b - c));
}

template <typename T>
T compute_volume(const std::array<Eigen::Vector<T, 3>, 3>& coords, const T& thickness) {
  const auto area = compute_area<T>(coords);
  return area * thickness;
}

template <typename T>
Eigen::Matrix<T, 3, 3> compute_rotation_matrix(const std::array<Eigen::Vector<T, 3>, 3>& localBasis) {

  Eigen::Matrix<T, 3, 3> m;
  m.template block<1, 3>(0, 0) = localBasis[0];
  m.template block<1, 3>(1, 0) = localBasis[1];
  m.template block<1, 3>(2, 0) = localBasis[2];

  return m;
}

/**
 * @brief Compute the in-plane coordinate coefficients a_i and b_i, which are
 * used to describe the normals and tangents.
 *
 * @param globalCoords The corner points of the triangle in the global
 * coordinate system
 * @param localBasis The local basis of the element
 * @return std::array<Eigen::Vector<T,3>, 2> {a_i, b_i}
 */
template <typename T>
std::array<Eigen::Vector<T, 3>, 2> compute_inplane_coefficients(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                                                const std::array<Eigen::Vector<T, 3>, 3>& localBasis) {

  const auto& e1 = localBasis[0];
  const auto& e2 = localBasis[1];

  const auto s1 = globalCoords[2] - globalCoords[1];
  const auto s2 = globalCoords[0] - globalCoords[2];
  const auto s3 = globalCoords[1] - globalCoords[0];

  Eigen::Vector<T, 3> a, b;
  a(0) = e1.dot(s1);
  a(1) = e1.dot(s2);
  a(2) = e1.dot(s3);
  b(0) = -e2.dot(s1);
  b(1) = -e2.dot(s2);
  b(2) = -e2.dot(s3);

  return {a, b};
}

template <typename T>
Eigen::Matrix<T, 3, 6> compute_membrane_strain_interpolation(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                                             const std::array<Eigen::Vector<T, 3>, 3>& localBasis,
                                                             const T& area) {

  const auto inplaneCoeffs     = Shell::compute_inplane_coefficients<T>(globalCoords, localBasis);
  const Eigen::Vector<T, 3> d1 = (1.0 / (2.0 * area)) * inplaneCoeffs[1];
  const Eigen::Vector<T, 3> d2 = (1.0 / (2.0 * area)) * inplaneCoeffs[0];

  Eigen::Matrix<T, 3, 6> bm = Eigen::Matrix<T, 3, 6>::Zero();

  bm.template block<1, 3>(0, 0) = d1.transpose();
  bm.template block<1, 3>(1, 3) = d2.transpose();
  bm.template block<1, 3>(2, 0) = d2.transpose();
  bm.template block<1, 3>(2, 3) = d1.transpose();

  return bm;
}

template <typename T>
Eigen::Matrix<T, 3, 6> compute_bending_strain_interpolation(const std::array<Eigen::Vector<T, 3>, 3>& localCoords,
                                                            const std::array<bool, 3>& flipRotations) {

  // based on Morleys paper
  // https://doi-org.proxy.findit.cvt.dk/10.1243/03093247V061020

  // extract corner coordinates -- note that we switch the 2nd and 3rd node, as
  // Morley defines the triangle nodes clock-wise
  const T x1 = localCoords[0](0);
  const T y1 = localCoords[0](1);
  const T x2 = localCoords[2](0);
  const T y2 = localCoords[2](1);
  const T x3 = localCoords[1](0);
  const T y3 = localCoords[1](1);

  const T b1 = y2 - y3;
  const T c1 = x3 - x2;

  const T b2 = y1 - y2;
  const T c2 = x2 - x1;

  const T b3 = y3 - y1;
  const T c3 = x1 - x3;

  /* NOTE, the above is equivalent to
  const auto inplane_coeffs = compute_inplane_coefficients(globalCoords, localBasis);
  const T c1 = inplane_coeffs[0](0); // a1 in FvK, c1 in Morley
  const T c2 = inplane_coeffs[0](1); // a2 in FvK, c2 in Morley
  const T c3 = inplane_coeffs[0](2); // a3 in FvK, c3 in Morley
  const T b1 = inplane_coeffs[1](0); // b1 in FvK, b1 in Morley
  const T b2 = inplane_coeffs[1](1); // b2 in FvK, b2 in Morley
  const T b3 = inplane_coeffs[1](2); // b3 in FvK, b3 in Morley
  */

  const T s12 = sqrt(b3 * b3 + c3 * c3);
  const T s31 = sqrt(b2 * b2 + c2 * c2);
  const T s23 = sqrt(b1 * b1 + c1 * c1);

  const T area = 0.5 * (c3 * b2 - c2 * b3);

  Eigen::Matrix<T, 3, 6> bb = Eigen::Matrix<T, 3, 6>::Zero();

  // out-of-plane displacement
  // note that we switch third and second node again, as these were switched for
  // the Morley formulation
  const T val1w                    = (b2 * c2) / (s31 * s31) - (b3 * c3) / (s12 * s12);
  const Eigen::Vector<T, 3> node1w = {val1w, -val1w, 2.0 * ((b2 * b2) / (s31 * s31) - (b3 * b3) / (s12 * s12))};

  const T val2w                    = (b3 * c3) / (s12 * s12) - (b1 * c1) / (s23 * s23);
  const Eigen::Vector<T, 3> node2w = {val2w, -val2w, 2.0 * ((b3 * b3) / (s12 * s12) - (b1 * b1) / (s23 * s23))};

  const T val3w                    = (b1 * c1) / (s23 * s23) - (b2 * c2) / (s31 * s31);
  const Eigen::Vector<T, 3> node3w = {val3w, -val3w, 2.0 * ((b1 * b1) / (s23 * s23) - (b2 * b2) / (s31 * s31))};

  bb.col(0) = node1w;
  bb.col(1) = node2w;
  bb.col(2) = node3w;

  // rotations - Note that the the nodes are switched compared to the marley
  // paper, to reflect the used node numbering.
  Eigen::Vector<T, 3> node12_rot = {-(b3 * b3) / s12, -(c3 * c3) / s12, (2.0 * b3 * c3) / s12};
  Eigen::Vector<T, 3> node23_rot = {-(b1 * b1) / s23, -(c1 * c1) / s23, (2.0 * b1 * c1) / s23};
  Eigen::Vector<T, 3> node31_rot = {-(b2 * b2) / s31, -(c2 * c2) / s31, (2.0 * b2 * c2) / s31};

  // flip the rotation definitien if necessary
  if (flipRotations[0])
    node12_rot *= -1.0;

  if (flipRotations[1])
    node23_rot *= -1.0;

  if (flipRotations[2])
    node31_rot *= -1.0;

  // apply the rotations.
  bb.col(3) = node12_rot;
  bb.col(4) = node23_rot;
  bb.col(5) = node31_rot;

  // scale with area
  bb *= 1.0 / area;

  return bb;
}

template <typename T>
Eigen::Matrix<T, 3, 3> compute_membrane_constitutive(const T& thickness, const T& youngs_module, const T& nu) {

  Eigen::Matrix<T, 3, 3> Emm = Eigen::Matrix<T, 3, 3>::Zero();

  Emm(0, 0) = 1;
  Emm(1, 1) = 1;
  Emm(0, 1) = nu;
  Emm(1, 0) = nu;
  Emm(2, 2) = 0.5 - 0.5 * nu;
  Emm *= ((youngs_module * thickness) / (1.0 - nu * nu));

  return Emm;
}

template <typename T>
Eigen::Matrix<T, 6, 6> compute_membrane_stiffness(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                                  const std::array<Eigen::Vector<T, 3>, 3>& localBasis,
                                                  const T& thickness, const T& youngs_module, const T& nu) {

  const T area   = compute_area<T>(globalCoords);
  const auto bm  = compute_membrane_strain_interpolation<T>(globalCoords, localBasis, area);
  const auto Emm = compute_membrane_constitutive(thickness, youngs_module, nu);

  return area * bm.transpose() * Emm * bm;
}

template <typename T>
Eigen::Matrix<T, 3, 3> compute_bending_constitutive(const T& thickness, const T& youngs_module, const T& nu) {

  Eigen::Matrix<T, 3, 3> Ebb = Eigen::Matrix<T, 3, 3>::Zero();

  Ebb(0, 0) = 1;
  Ebb(1, 1) = 1;
  Ebb(0, 1) = nu;
  Ebb(1, 0) = nu;
  Ebb(2, 2) = 0.5 - 0.5 * nu;
  Ebb *= (youngs_module * thickness * thickness * thickness) / (12.0 * (1.0 - nu * nu));

  return Ebb;
}

template <typename T>
Eigen::Matrix<T, 6, 6> compute_bending_stiffness(const std::array<Eigen::Vector<T, 3>, 3>& localCoords,
                                                 const std::array<bool, 3>& flipRotations, const T& thickness,
                                                 const T& youngs_module, const T& nu) {

  Eigen::Matrix<T, 3, 6> bt = compute_bending_strain_interpolation<T>(localCoords, flipRotations);
  const T area              = compute_area<T>(localCoords);
  const auto Ebb            = compute_bending_constitutive(thickness, youngs_module, nu);

  return area * bt.transpose() * Ebb * bt;
}

template <typename T>
Eigen::Matrix<T, 12, 12> compute_element_stiffness(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                                   const std::array<bool, 3>& flipRotations, const T& thickness,
                                                   const T& youngs_module, const T& nu) {

  const auto basis = compute_local_basis<T>(globalCoords);
  const auto Tn    = compute_rotation_matrix<T>(basis);

  std::array<Eigen::Vector<T, 3>, 3> localCoords;
  localCoords[0] = Tn * globalCoords[0];
  localCoords[1] = Tn * globalCoords[1];
  localCoords[2] = Tn * globalCoords[2];

  auto km = compute_membrane_stiffness<T>(globalCoords, basis, thickness, youngs_module, nu);
  auto kb = compute_bending_stiffness<T>(localCoords, flipRotations, thickness, youngs_module, nu);

  Eigen::Matrix<T, 12, 12> Te = Eigen::Matrix<T, 12, 12>::Identity();
  for (int t = 0; t < 3; t++) {
    Te(Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>), Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>)) = Tn;
  }
  // set stiffness matrix as (u1,u2,u3,v1,v2,v3,w1,w2,w3,r12,r23,r31)
  Eigen::Matrix<T, 12, 12> ke   = Eigen::Matrix<T, 12, 12>::Zero();
  ke.template block<6, 6>(0, 0) = km;
  ke.template block<6, 6>(6, 6) = kb;

  return Te.transpose() * ke * Te;
}

template <typename T>
std::array<T, 6> to_global_flat(const Eigen::Matrix<T, 3, 3>& Tn, const Eigen::Vector<T, 3>& stress) {

  Eigen::Matrix<T, 3, 3> Sb = Eigen::Matrix<T, 3, 3>::Zero();
  Sb(0, 0)                  = stress(0);
  Sb(1, 1)                  = stress(1);
  Sb(0, 1)                  = stress(2);
  Sb(1, 0)                  = stress(2);
  Sb                        = Tn.transpose() * Sb * Tn;
  return {Sb(0, 0), Sb(1, 1), Sb(2, 2), Sb(0, 1), Sb(0, 2), Sb(1, 2)};
}

template <typename T>
std::pair<std::array<T, 6>, std::array<T, 6>>
compute_element_stress(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords, const std::array<bool, 3>& flipRotations,
                       const T& thickness, const Eigen::Vector<T, 12>& u, const T& youngs_module, const T& nu) {

  const auto basis = compute_local_basis<T>(globalCoords);
  const auto Tn    = compute_rotation_matrix<T>(basis);
  const auto area  = compute_area<T>(globalCoords);

  std::array<Eigen::Vector<T, 3>, 3> localCoords;
  localCoords[0] = Tn * globalCoords[0];
  localCoords[1] = Tn * globalCoords[1];
  localCoords[2] = Tn * globalCoords[2];

  Eigen::Matrix<T, 12, 12> Te = Eigen::Matrix<T, 12, 12>::Identity();
  for (int t = 0; t < 3; t++) {
    Te(Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>), Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>)) = Tn;
  }

  Eigen::Vector<T, 12> u_local   = Te * u;
  Eigen::Vector<T, 6> u_membrane = u_local.template block<6, 1>(0, 0);
  Eigen::Vector<T, 6> u_bending  = u_local.template block<6, 1>(6, 0);

  const auto bm = Shell::compute_membrane_strain_interpolation(globalCoords, basis, area);
  const auto bb = Shell::compute_bending_strain_interpolation(localCoords, flipRotations);

  const auto Ebb = compute_bending_constitutive(thickness, youngs_module, nu);
  const auto Emm = compute_membrane_constitutive(thickness, youngs_module, nu);

  const Eigen::Vector<T, 3> membrane_stress = Emm * bm * u_membrane;
  const Eigen::Vector<T, 3> bending_stress  = Ebb * bb * u_bending;

  return {to_global_flat(Tn, membrane_stress), to_global_flat(Tn, bending_stress)};
}

template <typename T>
std::array<T, 2> compute_bending_and_membrane_compliance(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                                         const std::array<bool, 3>& flipRotations, const T& thickness,
                                                         Eigen::Vector<T, 12>& u, const T& youngs_module, const T& nu) {

  const auto basis = Shell::compute_local_basis<T>(globalCoords);
  const auto Tn    = Shell::compute_rotation_matrix<T>(basis);
  const auto area  = compute_area<T>(globalCoords);

  std::array<Eigen::Vector<T, 3>, 3> localCoords;
  localCoords[0] = Tn * globalCoords[0];
  localCoords[1] = Tn * globalCoords[1];
  localCoords[2] = Tn * globalCoords[2];

  const auto bm = Shell::compute_membrane_strain_interpolation(globalCoords, basis, area);
  const auto bb = Shell::compute_bending_strain_interpolation(localCoords, flipRotations);

  const auto Ebb = compute_bending_constitutive(thickness, youngs_module, nu);
  const auto Emm = compute_membrane_constitutive(thickness, youngs_module, nu);

  Eigen::Matrix<T, 12, 12> Te = Eigen::Matrix<T, 12, 12>::Identity();

  for (int t = 0; t < 3; t++) {
    Te(Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>), Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>)) = Tn;
  }

  // transform displacement to local coordinates.
  Eigen::Vector<T, 12> u_local   = Te * u;
  Eigen::Vector<T, 6> u_membrane = u_local.template block<6, 1>(0, 0);
  Eigen::Vector<T, 6> u_bending  = u_local.template block<6, 1>(6, 0);

  const Eigen::Vector<T, 3> strain_membrane = bm * u_membrane;
  const Eigen::Vector<T, 3> strain_bending  = bb * u_bending;

  const T membraneEnergy = strain_membrane.transpose() * Emm * strain_membrane;
  const T bendingEnergy  = strain_bending.transpose() * Ebb * strain_bending;

  return {area * membraneEnergy, area * bendingEnergy};
}

template <typename T>
T compute_compliance(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords, const std::array<bool, 3>& flipRotations,
                     const T& thickness, Eigen::Vector<T, 12>& u, const T& youngs_module, const T& nu) {

  const auto compliances =
      Shell::compute_bending_and_membrane_compliance<T>(globalCoords, flipRotations, thickness, u, youngs_module, nu);

  return compliances[0] + compliances[1];
}

template <typename T>
T compute_vm_stress(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords, const std::array<bool, 3>& flipRotations,
                    const T& thickness, Eigen::Vector<T, 12>& u, const T& youngs_module, const T& nu) {

  const auto basis = Shell::compute_local_basis<T>(globalCoords);
  const auto Tn    = Shell::compute_rotation_matrix<T>(basis);
  const auto area  = compute_area<T>(globalCoords);

  std::array<Eigen::Vector<T, 3>, 3> localCoords;
  localCoords[0] = Tn * globalCoords[0];
  localCoords[1] = Tn * globalCoords[1];
  localCoords[2] = Tn * globalCoords[2];

  const auto bm = Shell::compute_membrane_strain_interpolation(globalCoords, basis, area);
  const auto bb = Shell::compute_bending_strain_interpolation(localCoords, flipRotations);

  const auto Ebb = compute_bending_constitutive(thickness, youngs_module, nu);
  const auto Emm = compute_membrane_constitutive(thickness, youngs_module, nu);

  Eigen::Matrix<T, 12, 12> Te = Eigen::Matrix<T, 12, 12>::Identity();

  for (int t = 0; t < 3; t++) {
    Te(Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>), Eigen::seqN(t, Eigen::fix<3>, Eigen::fix<3>)) = Tn;
  }

  // transform displacement to local coordinates.
  Eigen::Vector<T, 12> u_local   = Te * u;
  Eigen::Vector<T, 6> u_membrane = u_local.template block<6, 1>(0, 0);
  Eigen::Vector<T, 6> u_bending  = u_local.template block<6, 1>(6, 0);

  const Eigen::Vector<T, 3> stress_membrane = Emm * bm * u_membrane;
  const Eigen::Vector<T, 3> stress_bending  = Ebb * bb * u_bending;

  const T& s_x  = stress_membrane(0);
  const T& s_y  = stress_membrane(1);
  const T& s_xy = stress_membrane(2);
  const T& s_z  = stress_bending(0);
  const T& s_zx = stress_bending(1);
  const T& s_zy = stress_bending(2);

  const T stress_normal = (s_x - s_y) * (s_x - s_y) + (s_x - s_z) * (s_x - s_z) + (s_z - s_y) * (s_z - s_y);
  const T stress_shear  = s_xy * s_xy + s_zx * s_zx + s_zy * s_zy;

  return sqrt(0.5 * stress_normal + 3.0 * stress_shear);
}

template <typename T>
Eigen::Vector<T, 3> compute_edge_traction(const std::array<Eigen::Vector<T, 3>, 2>& globalCoords,
                                          const Eigen::Vector<T, 3>& tractionForce) {

  const auto length = (globalCoords[0] - globalCoords[1]).norm();
  return (length * 0.5) * tractionForce;
}

template <typename T>
Eigen::Vector<T, 3> compute_element_traction(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                             const Eigen::Vector<T, 3>& tractionForce) {
  const auto area = compute_area<T>(globalCoords);
  return (area * (1.0 / 3.0)) * tractionForce;
}

template <typename T>
Eigen::Vector<T, 3> compute_element_pressure(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords,
                                             const T& pressure) {

  const auto basis = Shell::compute_local_basis<T>(globalCoords);

  // load is -pressure * normal
  const auto pressureLoad = -pressure * basis[2];

  return compute_element_traction<T>(globalCoords, pressureLoad);
}

template <typename T>
Eigen::Matrix<T, 3, 3> compute_geometry_mass(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords) {

  const auto area = compute_area<T>(globalCoords);

  const T v                = 1.0 / 3.0;
  Eigen::Matrix<T, 1, 3> N = {v, v, v};

  return area * (N.transpose() * N);
}

template <typename T>
Eigen::Matrix<T, 3, 3> compute_geometry_laplace(const std::array<Eigen::Vector<T, 3>, 3>& globalCoords) {

  const auto area  = compute_area<T>(globalCoords);
  const auto basis = Shell::compute_local_basis<T>(globalCoords);
  const auto coeff = compute_inplane_coefficients(globalCoords, basis);
  const auto& a    = coeff[0];
  const auto& b    = coeff[1];

  const T scale = 1.0 / (2.0 * area);

  Eigen::Matrix<T, 2, 3> B;
  B.template block<1, 3>(0, 0) = scale * b;
  B.template block<1, 3>(1, 0) = scale * a;

  return area * (B.transpose() * B);
}

} // namespace Shell