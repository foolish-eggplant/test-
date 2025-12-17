#include "HexElement.cuh"
#include "ModalBasis.cuh"
#include <cmath>

namespace zidingyi
{

__host__ __device__ static inline float EvaluateModal3D(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float tk = ModifiedA(k, rst.z);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = ModifiedA(j, rst.y);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float ri = ModifiedA(i, rst.x);
        value += coeffs[idx++] * ri * sj * tk;
      }
    }
  }
  return value;
}

__host__ __device__ static inline float EvaluateModal3D_dR(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float tk = ModifiedA(k, rst.z);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = ModifiedA(j, rst.y);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float riPrime = ModifiedAPrime(i, rst.x);
        value += coeffs[idx++] * riPrime * sj * tk;
      }
    }
  }
  return value;
}

__host__ __device__ static inline float EvaluateModal3D_dS(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float tk = ModifiedA(k, rst.z);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sjPrime = ModifiedAPrime(j, rst.y);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float ri = ModifiedA(i, rst.x);
        value += coeffs[idx++] * ri * sjPrime * tk;
      }
    }
  }
  return value;
}

__host__ __device__ static inline float EvaluateModal3D_dT(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float tkPrime = ModifiedAPrime(k, rst.z);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = ModifiedA(j, rst.y);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float ri = ModifiedA(i, rst.x);
        value += coeffs[idx++] * ri * sj * tkPrime;
      }
    }
  }
  return value;
}

__host__ __device__ static inline float ShapeWeight(int idx, float r, float s, float t)
{
  const float rSign = (idx & 1) ? 1.0f : -1.0f;
  const float sSign = (idx & 2) ? 1.0f : -1.0f;
  const float tSign = (idx & 4) ? 1.0f : -1.0f;
  return 0.125f * (1.0f + rSign * r) * (1.0f + sSign * s) * (1.0f + tSign * t);
}

__host__ __device__ float3 ReferenceToWorldTrilinear(
  const HexElementData& elem, const float3& rst)
{
  // Map XML/VTK vertex order -> ShapeWeight Z-order so weights apply to the
  // correct physical vertices.
  const int xmlToMathMap[8] = {0, 1, 3, 2, 4, 5, 7, 6};
  float3 result = make_float3(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < 8; ++i)
  {
    const int shapeIdx = xmlToMathMap[i];
    const float w = ShapeWeight(shapeIdx, rst.x, rst.y, rst.z);
    result.x += w * elem.vertices[i].x;
    result.y += w * elem.vertices[i].y;
    result.z += w * elem.vertices[i].z;
  }
  return result;
}

__host__ __device__ static inline float3 ReferenceToWorldHighOrder(
  const HexElementData& elem, const float3& rst)
{
  const float x = EvaluateModal3D(elem.geomCoefficients[0], elem.geomModes, rst);
  const float y = EvaluateModal3D(elem.geomCoefficients[1], elem.geomModes, rst);
  const float z = EvaluateModal3D(elem.geomCoefficients[2], elem.geomModes, rst);
  return make_float3(x, y, z);
}

__host__ __device__ float3 ReferenceToWorldHex(
  const HexElementData& elem, const float3& rst)
{
  if (HasHighOrderGeom(elem))
  {
    return ReferenceToWorldHighOrder(elem, rst);
  }
  return ReferenceToWorldTrilinear(elem, rst);
}

__host__ __device__ void JacobianTrilinear(
  const HexElementData& elem, const float3& rst, float J[9])
{
  // Map XML/VTK vertex order -> ShapeWeight Z-order so derivative signs match
  // the physical vertex positions.
  const int xmlToMathMap[8] = {0, 1, 3, 2, 4, 5, 7, 6};
  float3 dPhi_dr = make_float3(0.0f, 0.0f, 0.0f);
  float3 dPhi_ds = make_float3(0.0f, 0.0f, 0.0f);
  float3 dPhi_dt = make_float3(0.0f, 0.0f, 0.0f);

  for (int i = 0; i < 8; ++i)
  {
    const int shapeIdx = xmlToMathMap[i];
    const float rSign = (shapeIdx & 1) ? 1.0f : -1.0f;
    const float sSign = (shapeIdx & 2) ? 1.0f : -1.0f;
    const float tSign = (shapeIdx & 4) ? 1.0f : -1.0f;
    const float weight = 0.125f *
                         (1.0f + sSign * rst.y) *
                         (1.0f + tSign * rst.z);
    const float dN_dr = weight * rSign;
    const float dN_ds = 0.125f *
                        (1.0f + rSign * rst.x) *
                        (1.0f + tSign * rst.z) *
                        sSign;
    const float dN_dt = 0.125f *
                        (1.0f + rSign * rst.x) *
                        (1.0f + sSign * rst.y) *
                        tSign;

    dPhi_dr.x += dN_dr * elem.vertices[i].x;
    dPhi_dr.y += dN_dr * elem.vertices[i].y;
    dPhi_dr.z += dN_dr * elem.vertices[i].z;

    dPhi_ds.x += dN_ds * elem.vertices[i].x;
    dPhi_ds.y += dN_ds * elem.vertices[i].y;
    dPhi_ds.z += dN_ds * elem.vertices[i].z;

    dPhi_dt.x += dN_dt * elem.vertices[i].x;
    dPhi_dt.y += dN_dt * elem.vertices[i].y;
    dPhi_dt.z += dN_dt * elem.vertices[i].z;
  }

  J[0] = dPhi_dr.x;
  J[1] = dPhi_ds.x;
  J[2] = dPhi_dt.x;
  J[3] = dPhi_dr.y;
  J[4] = dPhi_ds.y;
  J[5] = dPhi_dt.y;
  J[6] = dPhi_dr.z;
  J[7] = dPhi_ds.z;
  J[8] = dPhi_dt.z;
}

__host__ __device__ void JacobianHighOrder(
  const HexElementData& elem, const float3& rst, float J[9])
{
  J[0] = EvaluateModal3D_dR(elem.geomCoefficients[0], elem.geomModes, rst);
  J[1] = EvaluateModal3D_dS(elem.geomCoefficients[0], elem.geomModes, rst);
  J[2] = EvaluateModal3D_dT(elem.geomCoefficients[0], elem.geomModes, rst);
  J[3] = EvaluateModal3D_dR(elem.geomCoefficients[1], elem.geomModes, rst);
  J[4] = EvaluateModal3D_dS(elem.geomCoefficients[1], elem.geomModes, rst);
  J[5] = EvaluateModal3D_dT(elem.geomCoefficients[1], elem.geomModes, rst);
  J[6] = EvaluateModal3D_dR(elem.geomCoefficients[2], elem.geomModes, rst);
  J[7] = EvaluateModal3D_dS(elem.geomCoefficients[2], elem.geomModes, rst);
  J[8] = EvaluateModal3D_dT(elem.geomCoefficients[2], elem.geomModes, rst);
}

__host__ __device__ void InvertJacobian(const float J[9], float inverse[9])
{
  const float det = J[0] * (J[4] * J[8] - J[5] * J[7]) -
                    J[1] * (J[3] * J[8] - J[5] * J[6]) +
                    J[2] * (J[3] * J[7] - J[4] * J[6]);

  const float invDet = 1.0f / det;

  inverse[0] = (J[4] * J[8] - J[5] * J[7]) * invDet;
  inverse[1] = (J[2] * J[7] - J[1] * J[8]) * invDet;
  inverse[2] = (J[1] * J[5] - J[2] * J[4]) * invDet;
  inverse[3] = (J[5] * J[6] - J[3] * J[8]) * invDet;
  inverse[4] = (J[0] * J[8] - J[2] * J[6]) * invDet;
  inverse[5] = (J[2] * J[3] - J[0] * J[5]) * invDet;
  inverse[6] = (J[3] * J[7] - J[4] * J[6]) * invDet;
  inverse[7] = (J[1] * J[6] - J[0] * J[7]) * invDet;
  inverse[8] = (J[0] * J[4] - J[1] * J[3]) * invDet;
}

__host__ __device__ bool WorldToReferenceNewton(
  const HexElementData& elem,
  const float3& worldPoint,
  float3& rst,
  int maxIterations,
  float tolerance)
{
  rst = make_float3(0.0f, 0.0f, 0.0f);
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    const float3 mapped = ReferenceToWorldHex(elem, rst);
    const float3 diff = make_float3(mapped.x - worldPoint.x,
                                    mapped.y - worldPoint.y,
                                    mapped.z - worldPoint.z);
    float J[9];
    if (HasHighOrderGeom(elem))
    {
      JacobianHighOrder(elem, rst, J);
    }
    else
    {
      JacobianTrilinear(elem, rst, J);
    }
    float invJ[9];
    InvertJacobian(J, invJ);

    const float deltaR =
      invJ[0] * diff.x + invJ[1] * diff.y + invJ[2] * diff.z;
    const float deltaS =
      invJ[3] * diff.x + invJ[4] * diff.y + invJ[5] * diff.z;
    const float deltaT =
      invJ[6] * diff.x + invJ[7] * diff.y + invJ[8] * diff.z;

    rst.x -= deltaR;
    rst.y -= deltaS;
    rst.z -= deltaT;

    if (fabsf(deltaR) < tolerance &&
        fabsf(deltaS) < tolerance &&
        fabsf(deltaT) < tolerance)
    {
      return true;
    }
  }
  return false;
}

__host__ __device__ float EvaluateModalField(
  const HexElementData& elem, const float3& rst)
{
  return EvaluateModal3D(elem.fieldCoefficients, elem.fieldModes, rst);
}

__host__ __device__ void EvaluateModalGradientReference(
  const HexElementData& elem, const float3& rst, float& du_dr, float& du_ds, float& du_dt)
{
  du_dr = du_ds = du_dt = 0.0f;
  const float* coeffs = elem.fieldCoefficients;
  const uint3 modes = elem.fieldModes;
  int idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float tk = ModifiedA(k, rst.z);
    const float tkPrime = ModifiedAPrime(k, rst.z);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = ModifiedA(j, rst.y);
      const float sjPrime = ModifiedAPrime(j, rst.y);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float ri = ModifiedA(i, rst.x);
        const float riPrime = ModifiedAPrime(i, rst.x);
        const float coeff = coeffs[idx++];
        du_dr += coeff * riPrime * sj * tk;
        du_ds += coeff * ri * sjPrime * tk;
        du_dt += coeff * ri * sj * tkPrime;
      }
    }
  }
}

__host__ __device__ float3 EvaluateModalGradientWorld(
  const HexElementData& elem, const float3& rst)
{
  float du_dr, du_ds, du_dt;
  EvaluateModalGradientReference(elem, rst, du_dr, du_ds, du_dt);
  float J[9];
  if (HasHighOrderGeom(elem))
  {
    JacobianHighOrder(elem, rst, J);
  }
  else
  {
    JacobianTrilinear(elem, rst, J);
  }
  float invJ[9];
  InvertJacobian(J, invJ);
  // grad_world = transpose(invJ) * grad_ref
  float3 grad;
  grad.x = invJ[0] * du_dr + invJ[3] * du_ds + invJ[6] * du_dt;
  grad.y = invJ[1] * du_dr + invJ[4] * du_ds + invJ[7] * du_dt;
  grad.z = invJ[2] * du_dr + invJ[5] * du_ds + invJ[8] * du_dt;
  return grad;
}
} // namespace zidingyi
