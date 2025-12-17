#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "RayTypes.cuh"

namespace zidingyi
{
struct PrismElementData
{
  float3 vertices[6];
  const float* fieldCoefficients{nullptr};
  uint3 fieldModes{0, 0, 0}; // modes.x, modes.y, modes.z (ModifiedB along z)

  const float* geomCoefficients[3]{nullptr, nullptr, nullptr};
  uint3 geomModes{0, 0, 0};
};

__host__ __device__ float3 ReferenceToWorldPrism(
  const PrismElementData& elem, const float3& rst);
__host__ __device__ void JacobianPrism(
  const PrismElementData& elem, const float3& rst, float J[9]);
__host__ __device__ bool WorldToReferenceNewtonPrism(
  const PrismElementData& elem,
  const float3& worldPoint,
  float3& rst,
  int maxIterations = 50,
  float tolerance = 1e-5f);
__host__ __device__ bool PrismContainsPoint(
  const PrismElementData& elem,
  const float3& worldPoint,
  float3* rstOut = nullptr,
  float tolerance = 1e-4f);
__host__ __device__ void ComputePrismAabb(
  const PrismElementData& elem, float3& minCorner, float3& maxCorner, int samplesPerAxis = 4);
__host__ __device__ bool IntersectRayWithPrism(
  const PrismElementData& elem,
  const Ray& ray,
  float tMin,
  float tMax,
  float& tHit,
  float3& rstHit,
  int maxIterations = 60,
  float tolerance = 1e-5f);

__host__ __device__ float EvaluatePrismField(
  const PrismElementData& elem, const float3& rst);
__host__ __device__ void EvaluatePrismGradientReference(
  const PrismElementData& elem,
  const float3& rst,
  float& du_dr,
  float& du_ds,
  float& du_dt);
__host__ __device__ float3 EvaluatePrismGradientWorld(
  const PrismElementData& elem, const float3& rst);
} // namespace zidingyi
