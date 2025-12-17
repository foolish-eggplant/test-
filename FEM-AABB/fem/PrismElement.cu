#include "PrismElement.cuh"
#include "ModalBasis.cuh"
#include "../geometry.h"
#include <cmath>
#include <float.h>

namespace  zidingyi
{
__host__ __device__ inline bool HasHighOrderGeom(const zidingyi::PrismElementData& elem)
{
  return elem.geomCoefficients[0] && elem.geomCoefficients[1] &&
         elem.geomCoefficients[2] && elem.geomModes.x > 0 && elem.geomModes.y > 0 &&
         elem.geomModes.z > 0;
}

__host__ __device__ inline float EvaluatePrismModal(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float ri = zidingyi::ModifiedA(i, rst.x);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = zidingyi::ModifiedA(j, rst.y);
      const unsigned int maxK = modes.z > i ? modes.z - i : 0;
      for (unsigned int k = 0; k < maxK; ++k)
      {
        const float tk = ModifiedB(i, k, rst.z);
        result += coeffs[idx++] * ri * sj * tk;
      }
    }
  }
  return result;
}

__host__ __device__ inline float EvaluatePrismModal_dR(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float riPrime = zidingyi::ModifiedAPrime(i, rst.x);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = zidingyi::ModifiedA(j, rst.y);
      const unsigned int maxK = modes.z > i ? modes.z - i : 0;
      for (unsigned int k = 0; k < maxK; ++k)
      {
        const float tk = ModifiedB(i, k, rst.z);
        result += coeffs[idx++] * riPrime * sj * tk;
      }
    }
  }
  return result;
}

__host__ __device__ inline float EvaluatePrismModal_dS(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float ri = zidingyi::ModifiedA(i, rst.x);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sjPrime = zidingyi::ModifiedAPrime(j, rst.y);
      const unsigned int maxK = modes.z > i ? modes.z - i : 0;
      for (unsigned int k = 0; k < maxK; ++k)
      {
        const float tk = ModifiedB(i, k, rst.z);
        result += coeffs[idx++] * ri * sjPrime * tk;
      }
    }
  }
  return result;
}

__host__ __device__ inline float EvaluatePrismModal_dT(
  const float* coeffs, uint3 modes, const float3& rst)
{
  if (!coeffs || modes.x == 0 || modes.y == 0 || modes.z == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float ri = zidingyi::ModifiedA(i, rst.x);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = zidingyi::ModifiedA(j, rst.y);
      const unsigned int maxK = modes.z > i ? modes.z - i : 0;
      for (unsigned int k = 0; k < maxK; ++k)
      {
        const float tkPrime = ModifiedBPrime(i, k, rst.z);
        result += coeffs[idx++] * ri * sj * tkPrime;
      }
    }
  }
  return result;
}

__host__ __device__ inline bool Solve3x3(const float A[9],
                                         const float3& b,
                                         float3& x)
{
  const float det =
    A[0] * (A[4] * A[8] - A[5] * A[7]) -
    A[1] * (A[3] * A[8] - A[5] * A[6]) +
    A[2] * (A[3] * A[7] - A[4] * A[6]);
  if (fabsf(det) < 1e-12f) return false;
  const float invDet = 1.0f / det;
  float invA[9];
  invA[0] = (A[4] * A[8] - A[5] * A[7]) * invDet;
  invA[1] = (A[2] * A[7] - A[1] * A[8]) * invDet;
  invA[2] = (A[1] * A[5] - A[2] * A[4]) * invDet;
  invA[3] = (A[5] * A[6] - A[3] * A[8]) * invDet;
  invA[4] = (A[0] * A[8] - A[2] * A[6]) * invDet;
  invA[5] = (A[2] * A[3] - A[0] * A[5]) * invDet;
  invA[6] = (A[3] * A[7] - A[4] * A[6]) * invDet;
  invA[7] = (A[1] * A[6] - A[0] * A[7]) * invDet;
  invA[8] = (A[0] * A[4] - A[1] * A[3]) * invDet;
  x.x = invA[0] * b.x + invA[1] * b.y + invA[2] * b.z;
  x.y = invA[3] * b.x + invA[4] * b.y + invA[5] * b.z;
  x.z = invA[6] * b.x + invA[7] * b.y + invA[8] * b.z;
  return true;
}

__host__ __device__ inline bool RayAabbIntersect(const zidingyi::Ray& ray,
                                                 const float3& minCorner,
                                                 const float3& maxCorner,
                                                 float tMin,
                                                 float tMax,
                                                 float& tEnter,
                                                 float& tExit)
{
  tEnter = tMin;
  tExit = tMax;
  for (int axis = 0; axis < 3; ++axis)
  {
    const float origin = axis == 0 ? ray.origin.x
                       : axis == 1 ? ray.origin.y
                                   : ray.origin.z;
    const float direction = axis == 0 ? ray.direction.x
                          : axis == 1 ? ray.direction.y
                                      : ray.direction.z;
    const float minValue = axis == 0 ? minCorner.x
                       : axis == 1 ? minCorner.y
                                   : minCorner.z;
    const float maxValue = axis == 0 ? maxCorner.x
                       : axis == 1 ? maxCorner.y
                                   : maxCorner.z;

    if (fabsf(direction) < 1e-8f)
    {
      if (origin < minValue || origin > maxValue) return false;
      continue;
    }

    float invDir = 1.0f / direction;
    float t0 = (minValue - origin) * invDir;
    float t1 = (maxValue - origin) * invDir;
    if (t0 > t1)
    {
      const float tmp = t0;
      t0 = t1;
      t1 = tmp;
    }
    tEnter = fmaxf(tEnter, t0);
    tExit = fminf(tExit, t1);
    if (tEnter > tExit) return false;
  }
  return tExit >= tEnter;
}

__host__ __device__ inline float ProjectPointToRayParameter(
  const zidingyi::Ray& ray, const float3& point)
{
  const float3 diff = make_float3(point.x - ray.origin.x,
                                  point.y - ray.origin.y,
                                  point.z - ray.origin.z);
  const float denom = dot(ray.direction, ray.direction);
  return denom > 0.0f ? dot(diff, ray.direction) / denom : 0.0f;
}
} // namespace

namespace zidingyi
{
__host__ __device__ float3 ReferenceToWorldLinearPrism(
  const PrismElementData& elem, const float3& rst)
{
  const float r = rst.x;
  const float s = rst.y;
  const float t = rst.z;

  const float t1 = (1.0f - r) * (1.0f - s) * (1.0f - t);
  const float t2 = (1.0f + r) * (1.0f - s) * (1.0f - t);
  const float t3 = (1.0f + r) * (1.0f + s) * (1.0f - t);
  const float t4 = (1.0f - r) * (1.0f + s) * (1.0f - t);
  const float t5 = (1.0f - s) * (1.0f + t);
  const float t6 = (1.0f + s) * (1.0f + t);

  const float scaleQuad = 0.125f;
  const float scaleTop = 0.25f;

  auto combine = [&](int idx, float weight)
  {
    return make_float3(weight * elem.vertices[idx].x,
                       weight * elem.vertices[idx].y,
                       weight * elem.vertices[idx].z);
  };

  float3 result = make_float3(0.0f, 0.0f, 0.0f);
  result += combine(0, scaleQuad * t1);
  result += combine(1, scaleQuad * t2);
  result += combine(2, scaleQuad * t3);
  result += combine(3, scaleQuad * t4);
  result += combine(4, scaleTop * t5);
  result += combine(5, scaleTop * t6);
  return result;
}

__host__ __device__ static inline float3 ReferenceToWorldHighOrderPrism(
  const PrismElementData& elem, const float3& rst)
{
  const float x = EvaluatePrismModal(elem.geomCoefficients[0], elem.geomModes, rst);
  const float y = EvaluatePrismModal(elem.geomCoefficients[1], elem.geomModes, rst);
  const float z = EvaluatePrismModal(elem.geomCoefficients[2], elem.geomModes, rst);
  return make_float3(x, y, z);
}

__host__ __device__ float3 ReferenceToWorldPrism(
  const PrismElementData& elem, const float3& rst)
{
  if (HasHighOrderGeom(elem))
  {
    return ReferenceToWorldHighOrderPrism(elem, rst);
  }
  return ReferenceToWorldLinearPrism(elem, rst);
}

__host__ __device__ void JacobianLinearPrism(
  const PrismElementData& elem, const float3& rst, float J[9])
{
  const float r = rst.x;
  const float s = rst.y;
  const float t = rst.z;

  const float eta00 = 1.0f - r;
  const float eta01 = 1.0f + r;
  const float eta10 = 1.0f - s;
  const float eta11 = 1.0f + s;
  const float eta20 = 1.0f - t;
  const float eta21 = 1.0f + t;

  float vX[6], vY[6], vZ[6];
  for (int i = 0; i < 6; ++i)
  {
    vX[i] = elem.vertices[i].x;
    vY[i] = elem.vertices[i].y;
    vZ[i] = elem.vertices[i].z;
  }

  for (int component = 0; component < 3; ++component)
  {
    const float* v = component == 0 ? vX : (component == 1 ? vY : vZ);
    const float v0c = v[0];
    const float v1c = v[1];
    const float v2c = v[2];
    const float v3c = v[3];
    const float v4c = v[4];
    const float v5c = v[5];

    const float dr =
      0.25f * (-eta10 * v0c + eta10 * v1c + eta11 * v2c - eta11 * v3c);
    const float ds =
      0.125f * (-eta00 * v0c - eta01 * v1c + eta01 * v2c + eta00 * v3c) *
        eta20 +
      0.25f * (-eta21 * v4c + eta21 * v5c);
    const float dt =
      0.125f *
        (-eta00 * eta10 * v0c - eta01 * eta10 * v1c - eta01 * eta11 * v2c -
         eta00 * eta11 * v3c) +
      0.25f * (eta10 * v4c + eta11 * v5c) + 0.5f * eta01 * dr;

    J[component * 3 + 0] = dr;
    J[component * 3 + 1] = ds;
    J[component * 3 + 2] = dt;
  }
}

__host__ __device__ static inline void JacobianHighOrderPrism(
  const PrismElementData& elem, const float3& rst, float J[9])
{
  J[0] = EvaluatePrismModal_dR(elem.geomCoefficients[0], elem.geomModes, rst);
  J[1] = EvaluatePrismModal_dS(elem.geomCoefficients[0], elem.geomModes, rst);
  J[2] = EvaluatePrismModal_dT(elem.geomCoefficients[0], elem.geomModes, rst);
  J[3] = EvaluatePrismModal_dR(elem.geomCoefficients[1], elem.geomModes, rst);
  J[4] = EvaluatePrismModal_dS(elem.geomCoefficients[1], elem.geomModes, rst);
  J[5] = EvaluatePrismModal_dT(elem.geomCoefficients[1], elem.geomModes, rst);
  J[6] = EvaluatePrismModal_dR(elem.geomCoefficients[2], elem.geomModes, rst);
  J[7] = EvaluatePrismModal_dS(elem.geomCoefficients[2], elem.geomModes, rst);
  J[8] = EvaluatePrismModal_dT(elem.geomCoefficients[2], elem.geomModes, rst);
}

__host__ __device__ void JacobianPrism(
  const PrismElementData& elem, const float3& rst, float J[9])
{
  if (HasHighOrderGeom(elem))
  {
    JacobianHighOrderPrism(elem, rst, J);
  }
  else
  {
    JacobianLinearPrism(elem, rst, J);
  }
}

__host__ __device__ static inline bool InvertPrismJacobian(
  const float J[9], float inverse[9])
{
  const float det = J[0] * (J[4] * J[8] - J[5] * J[7]) -
                    J[1] * (J[3] * J[8] - J[5] * J[6]) +
                    J[2] * (J[3] * J[7] - J[4] * J[6]);
  if (fabsf(det) < 1e-12f) return false;
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
  return true;
}

__host__ __device__ static inline bool SolveWithInitialGuess(
  const PrismElementData& elem,
  const float3& worldPoint,
  const float3& initialGuess,
  float3& rst,
  int maxIterations,
  float tolerance)
{
  rst = initialGuess;
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    const float3 mapped = ReferenceToWorldPrism(elem, rst);
    const float3 diff = mapped - worldPoint;
    float J[9], invJ[9];
    JacobianPrism(elem, rst, J);
    if (!InvertPrismJacobian(J, invJ)) return false;

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

__host__ __device__ bool WorldToReferenceNewtonPrism(
  const PrismElementData& elem,
  const float3& worldPoint,
  float3& rst,
  int maxIterations,
  float tolerance)
{
  return SolveWithInitialGuess(elem, worldPoint,
                               make_float3(0.0f, 0.0f, 0.0f),
                               rst, maxIterations, tolerance);
}

__host__ __device__ bool PrismContainsPoint(
  const PrismElementData& elem,
  const float3& worldPoint,
  float3* rstOut,
  float tolerance)
{
  const float3 seeds[] = {
    make_float3(0.0f, 0.0f, 0.0f),
    make_float3(-0.5f, -0.5f, -0.5f),
    make_float3(0.5f, -0.5f, -0.5f),
    make_float3(0.0f, 0.5f, -0.5f),
    make_float3(-0.5f, 0.0f, 0.5f),
    make_float3(0.5f, 0.0f, 0.5f),
    make_float3(0.0f, 0.5f, 0.5f)};

  for (const auto& seed : seeds)
  {
    float3 rst;
    if (!SolveWithInitialGuess(elem, worldPoint, seed, rst, 60, tolerance)) continue;
    if (rst.x < -1.05f || rst.x > 1.05f ||
        rst.y < -1.05f || rst.y > 1.05f ||
        rst.z < -1.05f || rst.z > 1.05f)
    {
      continue;
    }
    const float3 checkPoint = ReferenceToWorldPrism(elem, rst);
    const float3 diff = make_float3(fabsf(checkPoint.x - worldPoint.x),
                                    fabsf(checkPoint.y - worldPoint.y),
                                    fabsf(checkPoint.z - worldPoint.z));
    if (diff.x > 5e-4f || diff.y > 5e-4f || diff.z > 5e-4f) continue;
    if (rstOut) *rstOut = rst;
    return true;
  }
  return false;
}

__host__ __device__ void ComputePrismAabb(
  const PrismElementData& elem, float3& minCorner, float3& maxCorner, int samplesPerAxis)
{
  minCorner = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  maxCorner = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  auto expand = [&](const float3& p)
  {
    minCorner.x = fminf(minCorner.x, p.x);
    minCorner.y = fminf(minCorner.y, p.y);
    minCorner.z = fminf(minCorner.z, p.z);
    maxCorner.x = fmaxf(maxCorner.x, p.x);
    maxCorner.y = fmaxf(maxCorner.y, p.y);
    maxCorner.z = fmaxf(maxCorner.z, p.z);
  };

  for (int i = 0; i < 6; ++i) expand(elem.vertices[i]);

  const int steps = samplesPerAxis < 2 ? 2 : samplesPerAxis;
  for (int k = 0; k < steps; ++k)
  {
    const float w = -1.0f + 2.0f * static_cast<float>(k) / (steps - 1);
    for (int j = 0; j < steps; ++j)
    {
      const float v = -1.0f + 2.0f * static_cast<float>(j) / (steps - 1);
      for (int i = 0; i < steps; ++i)
      {
        const float u = -1.0f + 2.0f * static_cast<float>(i) / (steps - 1);
        const float3 p = ReferenceToWorldPrism(elem, make_float3(u, v, w));
        expand(p);
      }
    }
  }
}

__host__ __device__ bool IntersectRayWithPrism(
  const PrismElementData& elem,
  const Ray& ray,
  float tMin,
  float tMax,
  float& tHit,
  float3& rstHit,
  int maxIterations,
  float tolerance)
{
  float3 minCorner, maxCorner;
  ComputePrismAabb(elem, minCorner, maxCorner);

  float aabbEnter, aabbExit;
  if (!RayAabbIntersect(ray, minCorner, maxCorner, tMin, tMax, aabbEnter, aabbExit))
  {
    return false;
  }

  const float3 seedList[] = {
    make_float3(0.0f, 0.0f, 0.0f),
    make_float3(-0.5f, -0.5f, -0.5f),
    make_float3(0.5f, -0.5f, -0.5f),
    make_float3(0.0f, 0.5f, -0.5f),
    make_float3(-0.5f, 0.0f, 0.5f),
    make_float3(0.5f, 0.0f, 0.5f),
    make_float3(0.0f, 0.5f, 0.5f)};

  for (const auto& seed : seedList)
  {
    float3 rst = seed;
    float tau = aabbEnter;
    for (int iter = 0; iter < maxIterations; ++iter)
    {
      const float3 surfacePoint = ReferenceToWorldPrism(elem, rst);
      tau = ProjectPointToRayParameter(ray, surfacePoint);
      if (tau < aabbEnter || tau > aabbExit) break;
      const float3 rayPoint =
        make_float3(ray.origin.x + tau * ray.direction.x,
                    ray.origin.y + tau * ray.direction.y,
                    ray.origin.z + tau * ray.direction.z);
      const float3 residual =
        make_float3(surfacePoint.x - rayPoint.x,
                    surfacePoint.y - rayPoint.y,
                    surfacePoint.z - rayPoint.z);
      if (fmaxf(fabsf(residual.x),
                fmaxf(fabsf(residual.y), fabsf(residual.z))) < tolerance &&
          rst.x >= -1.05f && rst.x <= 1.05f &&
          rst.y >= -1.05f && rst.y <= 1.05f &&
          rst.z >= -1.05f && rst.z <= 1.05f)
      {
        tHit = tau;
        rstHit = rst;
        return true;
      }

      float J[9];
      JacobianPrism(elem, rst, J);
      float3 delta;
      float A[9] = {
        J[0], J[1], J[2],
        J[3], J[4], J[5],
        J[6], J[7], J[8]};
      if (!Solve3x3(A, residual, delta))
      {
        break;
      }

      rst.x -= delta.x;
      rst.y -= delta.y;
      rst.z -= delta.z;
    }
  }
  return false;
}

__host__ __device__ float EvaluatePrismField(
  const PrismElementData& elem, const float3& rst)
{
  return EvaluatePrismModal(elem.fieldCoefficients, elem.fieldModes, rst);
}

__host__ __device__ void EvaluatePrismGradientReference(
  const PrismElementData& elem,
  const float3& rst,
  float& du_dr,
  float& du_ds,
  float& du_dt)
{
  du_dr = du_ds = du_dt = 0.0f;
  const uint3 modes = elem.fieldModes;
  const float* coeffs = elem.fieldCoefficients;
  if (!coeffs) return;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float ri = ModifiedA(i, rst.x);
    const float riPrime = ModifiedAPrime(i, rst.x);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = ModifiedA(j, rst.y);
      const float sjPrime = ModifiedAPrime(j, rst.y);
      const unsigned int maxK = modes.z > i ? modes.z - i : 0;
      for (unsigned int k = 0; k < maxK; ++k)
      {
        const float tk = ModifiedB(i, k, rst.z);
        const float tkPrime = ModifiedBPrime(i, k, rst.z);
        const float coeff = coeffs[idx++];
        du_dr += coeff * riPrime * sj * tk;
        du_ds += coeff * ri * sjPrime * tk;
        du_dt += coeff * ri * sj * tkPrime;
      }
    }
  }
}

__host__ __device__ float3 EvaluatePrismGradientWorld(
  const PrismElementData& elem, const float3& rst)
{
  float du_dr, du_ds, du_dt;
  EvaluatePrismGradientReference(elem, rst, du_dr, du_ds, du_dt);
  float J[9], invJ[9];
  JacobianPrism(elem, rst, J);
  if (!InvertPrismJacobian(J, invJ))
  {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  float3 grad;
  grad.x = invJ[0] * du_dr + invJ[3] * du_ds + invJ[6] * du_dt;
  grad.y = invJ[1] * du_dr + invJ[4] * du_ds + invJ[7] * du_dt;
  grad.z = invJ[2] * du_dr + invJ[5] * du_ds + invJ[8] * du_dt;
  return grad;
}
} // namespace zidingyi
