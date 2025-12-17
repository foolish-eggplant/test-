#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace zidingyi
{
struct HexElementData
{
  float3 vertices[8];       // world-space corner vertices
  const float* fieldCoefficients{nullptr}; // modal coefficients for the scalar field
  uint3 fieldModes{0, 0, 0};              // number of modes in r/s/t (per ModifiedA basis)

  // Optional high-order geometry (x/y/z components)
  const float* geomCoefficients[3]{nullptr, nullptr, nullptr};
  uint3 geomModes{0, 0, 0};

  // Optional precomputed bounding box (world space)
  float3 aabbMin{0.0f, 0.0f, 0.0f};
  float3 aabbMax{0.0f, 0.0f, 0.0f};
};

// Basis utilities (Modified A polynomials used by ElVis / Nektar++)
__host__ __device__ float ModifiedA(unsigned int i, float x);
__host__ __device__ float ModifiedAPrime(unsigned int i, float x);

// Geometry
// Inline so device code in other translation units can see the body without a separate link step.
__host__ __device__ inline bool HasHighOrderGeom(const HexElementData& elem)
{
  return elem.geomCoefficients[0] && elem.geomCoefficients[1] &&
         elem.geomCoefficients[2] && elem.geomModes.x > 0 &&
         elem.geomModes.y > 0 && elem.geomModes.z > 0;
}
__host__ __device__ float EvaluateModal3D(const float* coeffs, uint3 modes, const float3& rst);
__host__ __device__ float EvaluateModal3D_dR(const float* coeffs, uint3 modes, const float3& rst);
__host__ __device__ float EvaluateModal3D_dS(const float* coeffs, uint3 modes, const float3& rst);
__host__ __device__ float EvaluateModal3D_dT(const float* coeffs, uint3 modes, const float3& rst);
__host__ __device__ float3 ReferenceToWorldHex(const HexElementData& elem, const float3& rst);
__host__ __device__ float3 ReferenceToWorldTrilinear(
  const HexElementData& elem, const float3& rst);
__host__ __device__ void JacobianTrilinear(
  const HexElementData& elem, const float3& rst, float J[9]);
__host__ __device__ void JacobianHighOrder(
  const HexElementData& elem, const float3& rst, float J[9]);
__host__ __device__ void InvertJacobian(const float J[9], float inverse[9]);
__host__ __device__ bool WorldToReferenceNewton(
  const HexElementData& elem,
  const float3& worldPoint,
  float3& rst,
  int maxIterations = 50,
  float tolerance = 1e-5f);

// Field evaluation
__host__ __device__ float EvaluateModalField(
  const HexElementData& elem, const float3& rst);
__host__ __device__ void EvaluateModalGradientReference(
  const HexElementData& elem, const float3& rst, float& du_dr, float& du_ds, float& du_dt);
__host__ __device__ float3 EvaluateModalGradientWorld(
  const HexElementData& elem, const float3& rst);
} // namespace zidingyi
