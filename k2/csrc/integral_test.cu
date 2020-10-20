/**
 * @brief Unittest for integral algorithm (this is really the application!)
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corp.   (Author: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <string>

#include "k2/csrc/integral.h"

namespace k2 {

TEST(ComputeIntegral, DebugExclusiveSum) {
  ContextPtr c = GetCudaContext();
  Array1<char> keep(c, 16777216 + 1, (char)0);
  keep = keep.Range(0, keep.Dim() - 1);

  Array1<int32_t> sum(c, keep.Dim() + 1);
  ExclusiveSum(keep, &sum);
  K2_CHECK_EQ(sum.Back(), 0);
}

TEST(ComputeIntegral, SinglePointAtOrigin) {

  // configuration with a single unit mass located at the origin.
  // should be the default (zero initialization) anyway...
  Configuration configuration;
  InitConfigurationDefault(&configuration);
  configuration.masses[0] = 1.0;

  ContextPtr c = GetCudaContext();
  for (int32_t i = 1; i < 2; i++) {
    double r = 10.0 * i;  // cube radius (==half edge length)
    double integral_error, integral_diff, abs_integral_diff;
    double integral = ComputeIntegral(c, configuration, r,
                                      1.0e-08,
                                      &integral_error, &integral_diff,
                                      &abs_integral_diff);
    K2_LOG(INFO) << "For r = " << r << ", one mass at origin, integral = "
                 << integral << " with (error,diff,abs-diff) "
                 << std::scientific << integral_error << "," << integral_diff
                 << "," << abs_integral_diff << std::fixed;
  }
}

TEST(ComputeIntegral, TwoEqualPointsSeparatedByOne) {

  // configuration with a single unit mass located at the origin.
  // should be the default (zero initialization) anyway...
  Configuration configuration;
  InitConfigurationDefault(&configuration);
  configuration.masses[0] = 0.5;
  configuration.masses[1] = 0.5;
  // make the points separated by 1.0.
  configuration.points[0].x[0] = 0.5;
  configuration.points[1].x[0] = -0.5;


  ContextPtr c = GetCudaContext();
  for (int32_t i = 1; i < 2; i++) {
    double r = 10.0 * i;  // cube radius (==half edge length)
    double integral_error, integral_diff, abs_integral_diff;
    double integral = ComputeIntegral(c, configuration, r,
                                      1.0e-08,
                                      &integral_error, &integral_diff,
                                      &abs_integral_diff);
    K2_LOG(INFO) << "For r = " << r << ", two masses separated near origin, integral = "
                 << integral << " with (error,diff,abs-diff) "
                 << std::scientific << integral_error << "," << integral_diff
                 << "," << abs_integral_diff << std::fixed;
  }
}

TEST(ComputeIntegral, TwoUnequalPointsSeparatedByOne) {

  // configuration with a single unit mass located at the origin.
  // should be the default (zero initialization) anyway...
  Configuration configuration;
  InitConfigurationDefault(&configuration);
  configuration.masses[0] = 0.25;
  configuration.masses[1] = 0.75;
  // make the points separated by 1.0.
  configuration.points[0].x[0] = 0.75;
  configuration.points[1].x[0] = -0.25;


  ContextPtr c = GetCudaContext();
  for (int32_t i = 1; i < 2; i++) {
    double r = 10.0 * i;  // cube radius (==half edge length)
    double integral_error, integral_diff, abs_integral_diff;
    double integral = ComputeIntegral(c, configuration, r,
                                      1.0e-08,
                                      &integral_error, &integral_diff,
                                      &abs_integral_diff);
    K2_LOG(INFO) << "For r = " << r << ", two different masses separated near origin, integral = "
                 << integral << " with (error,diff,abs-diff) "
                 << std::scientific << integral_error << "," << integral_diff
                 << "," << abs_integral_diff << std::fixed;
  }
}


TEST(ComputeIntegral, FourEqualPointsSeparatedByOne) {

  // configuration with a single unit mass located at the origin.
  // should be the default (zero initialization) anyway...
  Configuration configuration;
  InitConfigurationDefault(&configuration);

  // make the points separated by 1.0.
  for (int i = 0; i < 4; i++) {
    configuration.masses[i] = 0.25;
    for (int j = 0; j < 2; j++)
      configuration.points[i].x[j] = 0.5 * GetSign(i, j);
  }

  ContextPtr c = GetCudaContext();
  for (int32_t i = 1; i < 2; i++) {
    double r = 10.0 * i;  // cube radius (==half edge length)
    double integral_error, integral_diff, abs_integral_diff;
    double integral = ComputeIntegral(c, configuration, r,
                                      1.0e-08,
                                      &integral_error, &integral_diff,
                                      &abs_integral_diff);
    K2_LOG(INFO) << "For r = " << r << ", four masses separated near origin, integral = "
                 << integral << " with (error,diff,abs-diff) "
                 << std::scientific << integral_error << "," << integral_diff
                 << "," << abs_integral_diff << std::fixed;
  }
}

TEST(ComputeIntegral, FourPointsSimplex) {

  // configuration with a single unit mass located at the origin.
  // should be the default (zero initialization) anyway...
  Configuration configuration;
  InitConfigurationDefault(&configuration);

  // make the points separated by 1.0.
  for (int i = 0; i < 4; i++) {
    configuration.masses[i] = 0.25;
    if (i < 3)
      configuration.points[i].x[i] = 1.0;
  }

  ContextPtr c = GetCudaContext();
  for (int32_t i = 1; i < 2; i++) {
    double r = 10.0 * i;  // cube radius (==half edge length)
    double integral_error, integral_diff, abs_integral_diff;
    double integral = ComputeIntegral(c, configuration, r,
                                      1.0e-08,
                                      &integral_error, &integral_diff,
                                      &abs_integral_diff);
    K2_LOG(INFO) << "For r = " << r << ", four masses in simplex, integral = "
                 << integral << " with (error,diff,abs-diff) "
                 << std::scientific << integral_error << "," << integral_diff
                 << "," << abs_integral_diff << std::fixed;
  }
}



}  // namespace k2
