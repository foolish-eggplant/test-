#include "TriElement.cuh"
#include "ModalBasis.cuh"
#include "../geometry.h"
#include <cmath>
#include <float.h>

namespace
{
__host__ __device__ inline bool HasHighOrderGeom(
  const zidingyi::TriElementData& elem)
{
  return elem.geomCoefficients[0] && elem.geomCoefficients[1] &&
         elem.geomCoefficients[2] && elem.geomModes.x > 0 &&
         elem.geomModes.y > 0;
}

__host__ __device__ inline float2 ReferenceToBarycentric(const float2& rs)
{
  float u = 0.5f * (rs.x + 1.0f);
  float v = 0.5f * (rs.y + 1.0f);
  u = fminf(fmaxf(u, 0.0f), 1.0f);
  v = fminf(fmaxf(v, 0.0f), 1.0f);
  if (u + v > 1.0f)
  {
    const float excess = u + v - 1.0f;
    u -= 0.5f * excess;
    v -= 0.5f * excess;
    u = fmaxf(u, 0.0f);
    v = fmaxf(v, 0.0f);
  }
  return make_float2(u, v);
}

__host__ __device__ inline float3 ReferenceToWorldLinear(
  const zidingyi::TriElementData& elem, const float2& rs)
{
  const float2 bary = ReferenceToBarycentric(rs);
  const float w = fmaxf(0.0f, 1.0f - bary.x - bary.y);
  float3 result = make_float3(0.0f, 0.0f, 0.0f);
  result += w * elem.vertices[0];
  result += bary.x * elem.vertices[1];
  result += bary.y * elem.vertices[2];
  return result;
}

__host__ __device__ inline float EvaluateTriModal(
  const float* coeffs, uint2 modes, const float2& rs)
{
  if (!coeffs || modes.x == 0 || modes.y == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float ai = zidingyi::ModifiedA(i, rs.x);
    const unsigned int maxJ = modes.y > i ? modes.y - i : 0;
    for (unsigned int j = 0; j < maxJ; ++j)
    {
      const float bj = zidingyi::ModifiedB(i, j, rs.y);
      result += coeffs[idx++] * ai * bj;
    }
  }
  return result;
}

__host__ __device__ inline float EvaluateTriModal_dR(
  const float* coeffs, uint2 modes, const float2& rs)
{
  if (!coeffs || modes.x == 0 || modes.y == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float aiPrime = zidingyi::ModifiedAPrime(i, rs.x);
    const unsigned int maxJ = modes.y > i ? modes.y - i : 0;
    for (unsigned int j = 0; j < maxJ; ++j)
    {
      const float bj = zidingyi::ModifiedB(i, j, rs.y);
      result += coeffs[idx++] * aiPrime * bj;
    }
  }
  return result;
}

__host__ __device__ inline float EvaluateTriModal_dS(
  const float* coeffs, uint2 modes, const float2& rs)
{
  if (!coeffs || modes.x == 0 || modes.y == 0) return 0.0f;
  float result = 0.0f;
  int idx = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const float ai = zidingyi::ModifiedA(i, rs.x);
    const unsigned int maxJ = modes.y > i ? modes.y - i : 0;
    for (unsigned int j = 0; j < maxJ; ++j)
    {
      const float bjPrime = zidingyi::ModifiedBPrime(i, j, rs.y);
      result += coeffs[idx++] * ai * bjPrime;
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

__host__ __device__ inline float ProjectPointToRayParameter(
  const zidingyi::Ray& ray, const float3& point)
{
  const float3 diff = make_float3(point.x - ray.origin.x,
                                  point.y - ray.origin.y,
                                  point.z - ray.origin.z);
  const float denom = dot(ray.direction, ray.direction);
  return denom > 0.0f ? dot(diff, ray.direction) / denom : 0.0f;
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

__host__ __device__ inline bool SolveWithInitialGuess(
  const zidingyi::TriElementData& elem,
  const float3& worldPoint,
  const float2& initialGuess,
  float2& rs,
  int maxIterations,
  float tolerance)
{
  rs = initialGuess;
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    const float3 mapped = zidingyi::ReferenceToWorldTriangle(elem, rs);
    const float3 diff = make_float3(mapped.x - worldPoint.x,
                                    mapped.y - worldPoint.y,
                                    mapped.z - worldPoint.z);
    float3 dPhi_dr, dPhi_ds;
    zidingyi::SurfaceJacobianTriangle(elem, rs, dPhi_dr, dPhi_ds);
    float G00 = dot(dPhi_dr, dPhi_dr);
    float G01 = dot(dPhi_dr, dPhi_ds);
    float G11 = dot(dPhi_ds, dPhi_ds);
    const float det = G00 * G11 - G01 * G01;
    if (fabsf(det) < 1e-10f)
    {
      return false;
    }
    const float invDet = 1.0f / det;
    const float rhs0 = dot(dPhi_dr, diff);
    const float rhs1 = dot(dPhi_ds, diff);
    const float deltaR = invDet * (G11 * rhs0 - G01 * rhs1);
    const float deltaS = invDet * (-G01 * rhs0 + G00 * rhs1);

    rs.x -= deltaR;
    rs.y -= deltaS;

    if (fabsf(deltaR) < tolerance && fabsf(deltaS) < tolerance)
    {
      return true;
    }
  }
  return false;
}

__host__ __device__ inline bool InitializeTriangleIntersection(
  const zidingyi::TriElementData& elem,
  const zidingyi::Ray& ray,
  float tMin,
  float tMax,
  float& tGuess,
  float2& rsGuess)
{
  float3 minCorner, maxCorner;
  zidingyi::ComputeTriangleAabb(elem, minCorner, maxCorner);
  float aabbEnter, aabbExit;
  if (!RayAabbIntersect(ray, minCorner, maxCorner, tMin, tMax, aabbEnter, aabbExit))
  {
    return false;
  }

  const int samples = 8;
  for (int i = 0; i <= samples; ++i)
  {
    const float u = static_cast<float>(i) / samples;
    const float t = aabbEnter + (aabbExit - aabbEnter) * u;
    const float3 worldPoint =
      make_float3(ray.origin.x + t * ray.direction.x,
                  ray.origin.y + t * ray.direction.y,
                  ray.origin.z + t * ray.direction.z);
    float2 rs;
    if (zidingyi::TriangleContainsPoint(elem, worldPoint, &rs))
    {
      rsGuess = rs;
      tGuess = ProjectPointToRayParameter(ray, zidingyi::ReferenceToWorldTriangle(elem, rs));
      return true;
    }
  }

  const float3 edge0 = make_float3(elem.vertices[1].x - elem.vertices[0].x,
                                   elem.vertices[1].y - elem.vertices[0].y,
                                   elem.vertices[1].z - elem.vertices[0].z);
  const float3 edge1 = make_float3(elem.vertices[2].x - elem.vertices[0].x,
                                   elem.vertices[2].y - elem.vertices[0].y,
                                   elem.vertices[2].z - elem.vertices[0].z);
  const float3 normal = normalize(cross(edge0, edge1));
  const float denom = dot(normal, ray.direction);
  float t = 0.5f * (aabbEnter + aabbExit);
  if (fabsf(denom) > 1e-6f)
  {
    const float numerator =
      dot(normal,
          make_float3(elem.vertices[0].x - ray.origin.x,
                      elem.vertices[0].y - ray.origin.y,
                      elem.vertices[0].z - ray.origin.z));
    const float planeT = numerator / denom;
    if (planeT >= aabbEnter && planeT <= aabbExit)
    {
      t = planeT;
    }
  }
  const float3 planePoint =
    make_float3(ray.origin.x + t * ray.direction.x,
                ray.origin.y + t * ray.direction.y,
                ray.origin.z + t * ray.direction.z);
  float2 rs;
  if (zidingyi::TriangleContainsPoint(elem, planePoint, &rs))
  {
    rsGuess = rs;
  }
  else
  {
    rsGuess = make_float2(0.0f, 0.0f);
  }
  tGuess = ProjectPointToRayParameter(ray, zidingyi::ReferenceToWorldTriangle(elem, rsGuess));
  return true;
}
} // namespace

namespace zidingyi
{
__host__ __device__ float3 ReferenceToWorldTriangle(
  const TriElementData& elem, const float2& rs)
{
  if (HasHighOrderGeom(elem))
  {
    const float x = EvaluateTriModal(elem.geomCoefficients[0], elem.geomModes, rs);
    const float y = EvaluateTriModal(elem.geomCoefficients[1], elem.geomModes, rs);
    const float z = EvaluateTriModal(elem.geomCoefficients[2], elem.geomModes, rs);
    return make_float3(x, y, z);
  }
  return ReferenceToWorldLinear(elem, rs);
}

__host__ __device__ void SurfaceJacobianTriangle(
  const TriElementData& elem,
  const float2& rs,
  float3& dPhi_dr,
  float3& dPhi_ds)
{
  if (HasHighOrderGeom(elem))
  {
    dPhi_dr = make_float3(
      EvaluateTriModal_dR(elem.geomCoefficients[0], elem.geomModes, rs),
      EvaluateTriModal_dR(elem.geomCoefficients[1], elem.geomModes, rs),
      EvaluateTriModal_dR(elem.geomCoefficients[2], elem.geomModes, rs));
    dPhi_ds = make_float3(
      EvaluateTriModal_dS(elem.geomCoefficients[0], elem.geomModes, rs),
      EvaluateTriModal_dS(elem.geomCoefficients[1], elem.geomModes, rs),
      EvaluateTriModal_dS(elem.geomCoefficients[2], elem.geomModes, rs));
    return;
  }

  const float3 edge10 = make_float3(elem.vertices[1].x - elem.vertices[0].x,
                                    elem.vertices[1].y - elem.vertices[0].y,
                                    elem.vertices[1].z - elem.vertices[0].z);
  const float3 edge20 = make_float3(elem.vertices[2].x - elem.vertices[0].x,
                                    elem.vertices[2].y - elem.vertices[0].y,
                                    elem.vertices[2].z - elem.vertices[0].z);
  dPhi_dr = 0.5f * edge10;
  dPhi_ds = 0.5f * edge20;
}

__host__ __device__ bool WorldToReferenceNewtonTriangle(
  const TriElementData& elem,
  const float3& worldPoint,
  float2& rs,
  int maxIterations,
  float tolerance)
{
  const float2 seeds[] = {
    make_float2(0.0f, 0.0f),
    make_float2(-0.5f, -0.5f),
    make_float2(0.5f, -0.5f),
    make_float2(0.0f, 0.5f)};
  for (size_t i = 0; i < sizeof(seeds)/sizeof(seeds[0]); ++i)
  {
    if (SolveWithInitialGuess(elem, worldPoint, seeds[i], rs, maxIterations, tolerance))
    {
      return true;
    }
  }
  return false;
}

__host__ __device__ bool TriangleContainsPoint(
  const TriElementData& elem,
  const float3& worldPoint,
  float2* rsOut,
  float tolerance)
{
  float2 rs;
  if (!WorldToReferenceNewtonTriangle(elem, worldPoint, rs, 30, tolerance))
  {
    return false;
  }
  const float2 bary = ReferenceToBarycentric(rs);
  const float w = 1.0f - bary.x - bary.y;
  const bool inside = bary.x >= -1e-3f && bary.y >= -1e-3f && w >= -1e-3f;
  if (inside && rsOut)
  {
    *rsOut = rs;
  }
  return inside;
}

__host__ __device__ void ComputeTriangleAabb(
  const TriElementData& elem,
  float3& minCorner,
  float3& maxCorner,
  int samplesPerEdge)
{
  minCorner = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  maxCorner = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (int i = 0; i < 3; ++i)
  {
    const float3 v = elem.vertices[i];
    minCorner.x = fminf(minCorner.x, v.x);
    minCorner.y = fminf(minCorner.y, v.y);
    minCorner.z = fminf(minCorner.z, v.z);
    maxCorner.x = fmaxf(maxCorner.x, v.x);
    maxCorner.y = fmaxf(maxCorner.y, v.y);
    maxCorner.z = fmaxf(maxCorner.z, v.z);
  }

  if (!HasHighOrderGeom(elem)) return;

  for (int i = 0; i <= samplesPerEdge; ++i)
  {
    for (int j = 0; j <= samplesPerEdge - i; ++j)
    {
      const float u = static_cast<float>(i) / samplesPerEdge;
      const float v = static_cast<float>(j) / samplesPerEdge;
      const float2 rs = make_float2(2.0f * u - 1.0f, 2.0f * v - 1.0f);
      const float3 p = ReferenceToWorldTriangle(elem, rs);
      minCorner.x = fminf(minCorner.x, p.x);
      minCorner.y = fminf(minCorner.y, p.y);
      minCorner.z = fminf(minCorner.z, p.z);
      maxCorner.x = fmaxf(maxCorner.x, p.x);
      maxCorner.y = fmaxf(maxCorner.y, p.y);
      maxCorner.z = fmaxf(maxCorner.z, p.z);
    }
  }
}

__host__ __device__ bool IntersectRayWithTriangle(
  const TriElementData& elem,
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
  if (!InitializeTriangleIntersection(elem, ray, tMin, tMax, tGuess, rsGuess))
  {
    return false;
  }

  float t = tGuess;
  float2 rs = rsGuess;
  for (int iter = 0; iter < maxIterations; ++iter)
  {
    const float3 surfacePoint = ReferenceToWorldTriangle(elem, rs);
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
        t >= tMin && t <= tMax)
    {
      rsHit = rs;
      tHit = t;
      return true;
    }

    float3 dPhi_dr, dPhi_ds;
    SurfaceJacobianTriangle(elem, rs, dPhi_dr, dPhi_ds);
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
  }
  return false;
}

__host__ __device__ float EvaluateTriangleField(
  const TriElementData& elem,
  const float2& rs)
{
  return EvaluateTriModal(elem.fieldCoefficients, elem.fieldModes, rs);
}

__host__ __device__ void EvaluateTriangleGradientReference(
  const TriElementData& elem,
  const float2& rs,
  float& du_dr,
  float& du_ds)
{
  du_dr = EvaluateTriModal_dR(elem.fieldCoefficients, elem.fieldModes, rs);
  du_ds = EvaluateTriModal_dS(elem.fieldCoefficients, elem.fieldModes, rs);
}

__host__ __device__ float3 EvaluateTriangleGradientWorld(
  const TriElementData& elem,
  const float2& rs)
{
  float du_dr, du_ds;
  EvaluateTriangleGradientReference(elem, rs, du_dr, du_ds);
  float3 dPhi_dr, dPhi_ds;
  SurfaceJacobianTriangle(elem, rs, dPhi_dr, dPhi_ds);
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
} // namespace zidingyi
