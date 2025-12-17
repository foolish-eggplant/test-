#pragma once

#include <cuda_runtime.h>
#include "RayTypes.cuh"
#include <cstdint>

namespace zidingyi
{
struct QuadElementData
{
  float3 vertices[4];
  const float* fieldCoefficients{nullptr};
  uint2 fieldModes{0, 0};

  // Optional high-order geometry (x/y/z components)
  const float* geomCoefficients[3]{nullptr, nullptr, nullptr};
  uint2 geomModes{0, 0};
};

__host__ __device__ float3 ReferenceToWorldQuad(
  const QuadElementData& elem, const float2& rs);
__host__ __device__ void SurfaceJacobian(
  const QuadElementData& elem, const float2& rs, float3& dPhi_dr, float3& dPhi_ds);
__host__ __device__ bool WorldToReferenceNewtonQuad(
  const QuadElementData& elem,
  const float3& worldPoint,
  float2& rs,
  int maxIterations = 40,
  float tolerance = 1e-5f);
__host__ __device__ bool QuadContainsPoint(
  const QuadElementData& elem,
  const float3& worldPoint,
  float2* rsOut = nullptr,
  float tolerance = 1e-4f);
__host__ __device__ void ComputeQuadAabb(
  const QuadElementData& elem, float3& minCorner, float3& maxCorner, int samplesPerAxis = 4);

__host__ __device__ bool IntersectRayWithQuad(
  const QuadElementData& elem,
  const Ray& ray,
  float tMin,
  float tMax,
  float& tHit,
  float2& rsHit,
  int maxIterations = 40,
  float tolerance = 1e-5f);

__host__ __device__ float EvaluateQuadField(
  const QuadElementData& elem, const float2& rs);
__host__ __device__ void EvaluateQuadGradientReference(
  const QuadElementData& elem,
  const float2& rs,
  float& du_dr,
  float& du_ds);
__host__ __device__ float3 EvaluateQuadGradientWorld(
  const QuadElementData& elem, const float2& rs);
} // namespace zidingyi
