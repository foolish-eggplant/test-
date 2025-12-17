#pragma once

#include "HexElement.cuh"
#include "PrismElement.cuh"
#include "QuadElement.cuh"
#include "RayTypes.cuh"
#include <cuda_runtime.h>

namespace zidingyi
{

__host__ __device__ bool FindIsosurfaceOnRayHex(
  const HexElementData& elem,
  const Ray& ray,
  float tEnter,
  float tExit,
  float isoValue,
  float& tHit,
  float3& rstHit,
  float3& gradWorld,
  float tolerance = 1e-5f);

__host__ __device__ bool FindIsosurfaceOnRayPrism(
  const PrismElementData& elem,
  const Ray& ray,
  float tEnter,
  float tExit,
  float isoValue,
  float& tHit,
  float3& rstHit,
  float3& gradWorld,
  float tolerance = 1e-5f);

} // namespace zidingyi
