#include "QuadElement.cuh"
#include "ModalBasis.cuh"
#include "../geometry.h"
#include <cmath>
#include <float.h>

namespace zidingyi
{
__host__ __device__ inline bool HasHighOrderGeom(const zidingyi::QuadElementData& elem)
{
  return elem.geomCoefficients[0] && elem.geomCoefficients[1] &&
         elem.geomCoefficients[2] && elem.geomModes.x > 0 && elem.geomModes.y > 0;
}

__host__ __device__ inline float ShapeWeight(int idx, float r, float s)
{
  const float rSign = (idx & 1) ? 1.0f : -1.0f;
  const float sSign = (idx & 2) ? 1.0f : -1.0f;
  return 0.25f * (1.0f + rSign * r) * (1.0f + sSign * s);
}

__host__ __device__ inline float EvaluateModal2D(
  const float* coeffs, uint2 modes, const float2& rs)
{
  if (!coeffs) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int j = 0; j < modes.y; ++j)
  {
    const float sj = zidingyi::ModifiedA(j, rs.y);
    for (unsigned int i = 0; i < modes.x; ++i)
    {
      const float ri = zidingyi::ModifiedA(i, rs.x);
      value += coeffs[idx++] * ri * sj;
    }
  }
  return value;
}

__host__ __device__ inline float EvaluateModal2D_dR(
  const float* coeffs, uint2 modes, const float2& rs)
{
  if (!coeffs) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int j = 0; j < modes.y; ++j)
  {
    const float sj = zidingyi::ModifiedA(j, rs.y);
    for (unsigned int i = 0; i < modes.x; ++i)
    {
      const float riPrime = zidingyi::ModifiedAPrime(i, rs.x);
      value += coeffs[idx++] * riPrime * sj;
    }
  }
  return value;
}

__host__ __device__ inline float EvaluateModal2D_dS(
  const float* coeffs, uint2 modes, const float2& rs)
{
  if (!coeffs) return 0.0f;
  float value = 0.0f;
  int idx = 0;
  for (unsigned int j = 0; j < modes.y; ++j)
  {
    const float sjPrime = zidingyi::ModifiedAPrime(j, rs.y);
    for (unsigned int i = 0; i < modes.x; ++i)
    {
      const float ri = zidingyi::ModifiedA(i, rs.x);
      value += coeffs[idx++] * ri * sjPrime;
    }
  }
  return value;
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
} // namespace

namespace zidingyi
{
__host__ __device__ static inline float3 ReferenceToWorldBilinear(
  const QuadElementData& elem, const float2& rs)
{
  float3 result = make_float3(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < 4; ++i)
  {
    const float w = ShapeWeight(i, rs.x, rs.y);
    result.x += w * elem.vertices[i].x;
    result.y += w * elem.vertices[i].y;
    result.z += w * elem.vertices[i].z;
  }
  return result;
}

__host__ __device__ static inline float3 ReferenceToWorldHighOrder(
  const QuadElementData& elem, const float2& rs)
{
  const float x = EvaluateModal2D(elem.geomCoefficients[0], elem.geomModes, rs);
  const float y = EvaluateModal2D(elem.geomCoefficients[1], elem.geomModes, rs);
  const float z = EvaluateModal2D(elem.geomCoefficients[2], elem.geomModes, rs);
  return make_float3(x, y, z);
}

__host__ __device__ float3 ReferenceToWorldQuad(
  const QuadElementData& elem, const float2& rs)
{
  if (HasHighOrderGeom(elem))
  {
    return ReferenceToWorldHighOrder(elem, rs);
  }
  return ReferenceToWorldBilinear(elem, rs);
}

__host__ __device__ void SurfaceJacobian(
  const QuadElementData& elem,
  const float2& rs,
  float3& dPhi_dr,
  float3& dPhi_ds)
{
  if (HasHighOrderGeom(elem))
  {
    dPhi_dr = make_float3(
      EvaluateModal2D_dR(elem.geomCoefficients[0], elem.geomModes, rs),
      EvaluateModal2D_dR(elem.geomCoefficients[1], elem.geomModes, rs),
      EvaluateModal2D_dR(elem.geomCoefficients[2], elem.geomModes, rs));
    dPhi_ds = make_float3(
      EvaluateModal2D_dS(elem.geomCoefficients[0], elem.geomModes, rs),
      EvaluateModal2D_dS(elem.geomCoefficients[1], elem.geomModes, rs),
      EvaluateModal2D_dS(elem.geomCoefficients[2], elem.geomModes, rs));
    return;
  }

  float3 dR = make_float3(0.0f, 0.0f, 0.0f);
  float3 dS = make_float3(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < 4; ++i)
  {
    const float rSign = (i & 1) ? 1.0f : -1.0f;
    const float sSign = (i & 2) ? 1.0f : -1.0f;
    const float dN_dr = 0.25f * rSign * (1.0f + sSign * rs.y);
    const float dN_ds = 0.25f * sSign * (1.0f + rSign * rs.x);
    dR.x += dN_dr * elem.vertices[i].x;
    dR.y += dN_dr * elem.vertices[i].y;
    dR.z += dN_dr * elem.vertices[i].z;
    dS.x += dN_ds * elem.vertices[i].x;
    dS.y += dN_ds * elem.vertices[i].y;
    dS.z += dN_ds * elem.vertices[i].z;
  }
  dPhi_dr = dR;
  dPhi_ds = dS;
}

__host__ __device__ static inline bool SolveWithInitialGuess(
  const QuadElementData& elem,
  const float3& worldPoint,
  const float2& initialGuess,
  float2& rs,
  int maxIterations,
  float tolerance)
{
  rs = initialGuess;
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    const float3 mapped = ReferenceToWorldQuad(elem, rs);
    const float3 diff = mapped - worldPoint;
    float3 dPhi_dr, dPhi_ds;
    SurfaceJacobian(elem, rs, dPhi_dr, dPhi_ds);
    float G00 = dot(dPhi_dr, dPhi_dr);
    float G01 = dot(dPhi_dr, dPhi_ds);
    float G11 = dot(dPhi_ds, dPhi_ds);
    const float rhs0 = dot(dPhi_dr, diff);
    const float rhs1 = dot(dPhi_ds, diff);
    const float det = G00 * G11 - G01 * G01;
    if (fabsf(det) < 1e-10f)
    {
      return false;
    }
    const float invDet = 1.0f / det;
    const float deltaR = invDet * (G11 * rhs0 - G01 * rhs1);
    const float deltaS = invDet * (-G01 * rhs0 + G00 * rhs1);

    rs.x -= deltaR;
    rs.y -= deltaS;

    if (fabsf(deltaR) < tolerance && fabsf(deltaS) < tolerance) return true;
  }
  return false;
}

__host__ __device__ bool WorldToReferenceNewtonQuad(
  const QuadElementData& elem,
  const float3& worldPoint,
  float2& rs,
  int maxIterations,
  float tolerance)
{
  return SolveWithInitialGuess(elem, worldPoint, make_float2(0.0f, 0.0f), rs,
                               maxIterations, tolerance);
}

__host__ __device__ bool QuadContainsPoint(
  const QuadElementData& elem,
  const float3& worldPoint,
  float2* rsOut,
  float tolerance)
{
  const float2 seeds[] = {
    make_float2(0.0f, 0.0f),  make_float2(-0.5f, -0.5f),
    make_float2(0.5f, -0.5f), make_float2(-0.5f, 0.5f),
    make_float2(0.5f, 0.5f),  make_float2(-0.9f, -0.9f),
    make_float2(0.9f, -0.9f), make_float2(-0.9f, 0.9f),
    make_float2(0.9f, 0.9f)};

  for (const auto& seed : seeds)
  {
    float2 rst;
    if (!SolveWithInitialGuess(elem, worldPoint, seed, rst, 40, tolerance)) continue;
    if (rst.x < -1.05f || rst.x > 1.05f || rst.y < -1.05f || rst.y > 1.05f)
      continue;
    const float3 checkPoint = ReferenceToWorldQuad(elem, rst);
    const float3 diff = make_float3(fabsf(checkPoint.x - worldPoint.x),
                                    fabsf(checkPoint.y - worldPoint.y),
                                    fabsf(checkPoint.z - worldPoint.z));
    if (diff.x > 5e-4f || diff.y > 5e-4f || diff.z > 5e-4f) continue;
    if (rsOut) *rsOut = rst;
    return true;
  }
  return false;
}

__host__ __device__ void ComputeQuadAabb(
  const QuadElementData& elem, float3& minCorner, float3& maxCorner, int samplesPerAxis)
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

  for (int i = 0; i < 4; ++i) expand(elem.vertices[i]);

  const int steps = samplesPerAxis < 2 ? 2 : samplesPerAxis;
  for (int j = 0; j < steps; ++j)
  {
    const float v = -1.0f + 2.0f * static_cast<float>(j) / (steps - 1);
    for (int i = 0; i < steps; ++i)
    {
      const float u = -1.0f + 2.0f * static_cast<float>(i) / (steps - 1);
      const float3 p = ReferenceToWorldQuad(elem, make_float2(u, v));
      expand(p);
    }
  }
}

__host__ __device__ static inline bool RayAabbIntersect(const Ray& ray,
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

__host__ __device__ float EvaluateQuadField(
  const QuadElementData& elem, const float2& rs)
{
  return EvaluateModal2D(elem.fieldCoefficients, elem.fieldModes, rs);
}

__host__ __device__ void EvaluateQuadGradientReference(
  const QuadElementData& elem,
  const float2& rs,
  float& du_dr,
  float& du_ds)
{
  const float* coeffs = elem.fieldCoefficients;
  const uint2 modes = elem.fieldModes;
  du_dr = du_ds = 0.0f;
  if (!coeffs) return;
  int idx = 0;
  for (unsigned int j = 0; j < modes.y; ++j)
  {
    const float sj = ModifiedA(j, rs.y);
    const float sjPrime = ModifiedAPrime(j, rs.y);
    for (unsigned int i = 0; i < modes.x; ++i)
    {
      const float ri = ModifiedA(i, rs.x);
      const float riPrime = ModifiedAPrime(i, rs.x);
      const float coeff = coeffs[idx++];
      du_dr += coeff * riPrime * sj;
      du_ds += coeff * ri * sjPrime;
    }
  }
}

__host__ __device__ float3 EvaluateQuadGradientWorld(
  const QuadElementData& elem, const float2& rs)
{
  float du_dr, du_ds;
  EvaluateQuadGradientReference(elem, rs, du_dr, du_ds);
  float3 dPhi_dr, dPhi_ds;
  SurfaceJacobian(elem, rs, dPhi_dr, dPhi_ds);
  float G00 = dot(dPhi_dr, dPhi_dr);
  float G01 = dot(dPhi_dr, dPhi_ds);
  float G11 = dot(dPhi_ds, dPhi_ds);
  const float det = G00 * G11 - G01 * G01;
  if (fabsf(det) < 1e-10f)
  {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  const float invDet = 1.0f / det;
  const float alpha = invDet * (G11 * du_dr - G01 * du_ds);
  const float beta = invDet * (-G01 * du_dr + G00 * du_ds);
  float3 grad;
  grad.x = alpha * dPhi_dr.x + beta * dPhi_ds.x;
  grad.y = alpha * dPhi_dr.y + beta * dPhi_ds.y;
  grad.z = alpha * dPhi_dr.z + beta * dPhi_ds.z;
  return grad;
}

__host__ __device__ static inline float ProjectPointToRayParameter(
  const Ray& ray, const float3& point)
{
  const float3 diff = make_float3(point.x - ray.origin.x,
                                  point.y - ray.origin.y,
                                  point.z - ray.origin.z);
  const float denom = dot(ray.direction, ray.direction);
  return denom > 0.0f ? dot(diff, ray.direction) / denom : 0.0f;
}

__host__ __device__ static inline bool InitializeQuadIntersection(
  const QuadElementData& elem,
  const Ray& ray,
  float tMin,
  float tMax,
  float& tGuess,
  float2& rsGuess)
{
  float3 minCorner, maxCorner;
  ComputeQuadAabb(elem, minCorner, maxCorner);

  float aabbEnter, aabbExit;
  if (!RayAabbIntersect(ray, minCorner, maxCorner, tMin, tMax, aabbEnter, aabbExit))
  {
    return false;
  }

  const int samples = 10;
  for (int i = 0; i <= samples; ++i)
  {
    const float u = static_cast<float>(i) / samples;
    const float t = aabbEnter + (aabbExit - aabbEnter) * u;
    const float3 worldPoint =
      make_float3(ray.origin.x + t * ray.direction.x,
                  ray.origin.y + t * ray.direction.y,
                  ray.origin.z + t * ray.direction.z);
    float2 rs;
    if (QuadContainsPoint(elem, worldPoint, &rs))
    {
      tGuess = ProjectPointToRayParameter(ray, ReferenceToWorldQuad(elem, rs));
      rsGuess = rs;
      return true;
    }
  }

  const float3 v0 = elem.vertices[0];
  const float3 v1 = elem.vertices[1];
  const float3 v3 = elem.vertices[3];
  const float3 normal =
    normalize(cross(make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z),
                    make_float3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z)));
  const float denom = dot(normal, ray.direction);
  if (fabsf(denom) > 1e-6f)
  {
    const float numerator =
      dot(normal,
          make_float3(v0.x - ray.origin.x, v0.y - ray.origin.y, v0.z - ray.origin.z));
    float t = numerator / denom;
    if (t < aabbEnter || t > aabbExit)
    {
      t = 0.5f * (aabbEnter + aabbExit);
    }
    const float3 planePoint =
      make_float3(ray.origin.x + t * ray.direction.x,
                  ray.origin.y + t * ray.direction.y,
                  ray.origin.z + t * ray.direction.z);
    if (QuadContainsPoint(elem, planePoint, &rsGuess))
    {
      tGuess = ProjectPointToRayParameter(ray, ReferenceToWorldQuad(elem, rsGuess));
      return true;
    }
  }

  rsGuess = make_float2(0.0f, 0.0f);
  tGuess = 0.5f * (aabbEnter + aabbExit);
  return true;
}

__host__ __device__ bool IntersectRayWithQuad(
  const QuadElementData& elem,
  const Ray& ray,
  float tMin,
  float tMax,
  float& tHit,
  float2& rsHit,
  int maxIterations,
  float tolerance)
{
  float tGuess;
  float2 rsGuess;
  if (!InitializeQuadIntersection(elem, ray, tMin, tMax, tGuess, rsGuess))
  {
    return false;
  }

  float t = tGuess;
  float2 rs = rsGuess;
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    const float3 surfacePoint = ReferenceToWorldQuad(elem, rs);
    const float3 rayPoint =
      make_float3(ray.origin.x + t * ray.direction.x,
                  ray.origin.y + t * ray.direction.y,
                  ray.origin.z + t * ray.direction.z);
    const float3 residual =
      make_float3(rayPoint.x - surfacePoint.x,
                  rayPoint.y - surfacePoint.y,
                  rayPoint.z - surfacePoint.z);
    if (fmaxf(fabsf(residual.x),
              fmaxf(fabsf(residual.y), fabsf(residual.z))) < tolerance &&
        rs.x >= -1.02f && rs.x <= 1.02f &&
        rs.y >= -1.02f && rs.y <= 1.02f &&
        t >= tMin && t <= tMax)
    {
      tHit = t;
      rsHit = rs;
      return true;
    }

    float3 dPhi_dr, dPhi_ds;
    SurfaceJacobian(elem, rs, dPhi_dr, dPhi_ds);

    float A[9] = {
      -dPhi_dr.x, -dPhi_ds.x, ray.direction.x,
      -dPhi_dr.y, -dPhi_ds.y, ray.direction.y,
      -dPhi_dr.z, -dPhi_ds.z, ray.direction.z};
    float3 b = make_float3(-residual.x, -residual.y, -residual.z);
    float3 delta;
    if (!Solve3x3(A, b, delta))
    {
      return false;
    }

    rs.x += delta.x;
    rs.y += delta.y;
    t += delta.z;
    if (t < tMin || t > tMax) return false;
  }
  return false;
}
} // namespace zidingyi
