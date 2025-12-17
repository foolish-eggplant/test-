#pragma once

#include <cuda_runtime.h>
#include "RayTypes.cuh"
#include <cstdint>

namespace zidingyi
{
struct TriElementData
{
  float3 vertices[3];
  const float* fieldCoefficients{nullptr};
  uint2 fieldModes{0, 0};

  const float* geomCoefficients[3]{nullptr, nullptr, nullptr};
  uint2 geomModes{0, 0};
};

__host__ __device__ float3 ReferenceToWorldTriangle(
  const TriElementData& elem, const float2& rs);
__host__ __device__ void SurfaceJacobianTriangle(
  const TriElementData& elem,
  const float2& rs,
  float3& dPhi_dr,
  float3& dPhi_ds);
__host__ __device__ bool WorldToReferenceNewtonTriangle(
  const TriElementData& elem,
  const float3& worldPoint,
  float2& rs,
  int maxIterations = 40,
  float tolerance = 1e-5f);
__host__ __device__ bool TriangleContainsPoint(
  const TriElementData& elem,
  const float3& worldPoint,
  float2* rsOut = nullptr,
  float tolerance = 1e-4f);
__host__ __device__ void ComputeTriangleAabb(
  const TriElementData& elem,
  float3& minCorner,
  float3& maxCorner,
  int samplesPerEdge = 6);
// Backward-compatible alias used in BVH / raytracer.
__host__ __device__ inline void ComputeTriAabb(
  const TriElementData& elem,
  float3& minCorner,
  float3& maxCorner,
  int samplesPerEdge = 6)
{
  ComputeTriangleAabb(elem, minCorner, maxCorner, samplesPerEdge);
}
__host__ __device__ bool IntersectRayWithTriangle(
  const TriElementData& elem,
  const Ray& ray,
  float tMin,
  float tMax,
  float& tHit,
  float2& rsHit,
  int maxIterations = 40,
  float tolerance = 1e-5f);

__host__ __device__ float EvaluateTriangleField(
  const TriElementData& elem,
  const float2& rs);
__host__ __device__ void EvaluateTriangleGradientReference(
  const TriElementData& elem,
  const float2& rs,
  float& du_dr,
  float& du_ds);
__host__ __device__ float3 EvaluateTriangleGradientWorld(
  const TriElementData& elem,
  const float2& rs);
} // namespace zidingyi
