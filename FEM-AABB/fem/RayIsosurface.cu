#include "RayIsosurface.cuh"
#include "HexElement.cuh"
#include "PrismElement.cuh"
#include <cmath>

namespace zidingyi
{
namespace
{
struct HexOps
{
  const HexElementData& elem;
  __host__ __device__ bool Solve(const float3& world, float3& rst) const
  {
    return WorldToReferenceNewton(elem, world, rst);
  }
  __host__ __device__ float Value(const float3& rst) const
  {
    return EvaluateModalField(elem, rst);
  }
  __host__ __device__ float3 Gradient(const float3& rst) const
  {
    return EvaluateModalGradientWorld(elem, rst);
  }
};

struct PrismOps
{
  const PrismElementData& elem;
  __host__ __device__ bool Solve(const float3& world, float3& rst) const
  {
    return WorldToReferenceNewtonPrism(elem, world, rst);
  }
  __host__ __device__ float Value(const float3& rst) const
  {
    return EvaluatePrismField(elem, rst);
  }
  __host__ __device__ float3 Gradient(const float3& rst) const
  {
    return EvaluatePrismGradientWorld(elem, rst);
  }
};

template <typename Ops>
__host__ __device__ bool EvaluateAlongRay(const Ops& ops,
                                          const Ray& ray,
                                          float t,
                                          float iso,
                                          float3& rst,
                                          float& fValue,
                                          float3& grad,
                                          float& derivative)
{
  const float3 worldPoint = make_float3(ray.origin.x + t * ray.direction.x,
                                        ray.origin.y + t * ray.direction.y,
                                        ray.origin.z + t * ray.direction.z);
  if (!ops.Solve(worldPoint, rst)) return false;
  const float u = ops.Value(rst);
  grad = ops.Gradient(rst);
  fValue = u - iso;
  derivative = grad.x * ray.direction.x +
               grad.y * ray.direction.y +
               grad.z * ray.direction.z;
  return true;
}

template <typename Ops>
__host__ __device__ bool FindIsosurfaceImpl(const Ops& ops,
                                            const Ray& ray,
                                            float tEnter,
                                            float tExit,
                                            float isoValue,
                                            float& tHit,
                                            float3& rstHit,
                                            float3& gradWorld,
                                            float tolerance)
{
  constexpr int kSamples = 16;
  float prevT = tEnter;
  float prevVal = 0.0f;
  float3 prevRst = make_float3(0.0f, 0.0f, 0.0f);
  float3 prevGrad = make_float3(0.0f, 0.0f, 0.0f);
  float prevDeriv = 0.0f;
  bool havePrev = false;

  for (int i = 0; i <= kSamples; ++i)
  {
    const float u = static_cast<float>(i) / kSamples;
    const float t = tEnter + (tExit - tEnter) * u;
    float value, deriv;
    float3 rst, grad;
    if (!EvaluateAlongRay(ops, ray, t, isoValue, rst, value, grad, deriv))
    {
      continue;
    }

    if (!havePrev)
    {
      prevT = t;
      prevVal = value;
      prevRst = rst;
      prevGrad = grad;
      prevDeriv = deriv;
      havePrev = true;
      continue;
    }

    if (prevVal * value <= 0.0f)
    {
      // bracket found between prevT and t
      float leftT = prevT;
      float leftVal = prevVal;
      float3 leftRst = prevRst;
      float3 leftGrad = prevGrad;

      float rightT = t;
      float rightVal = value;
      float3 rightRst = rst;
      float3 rightGrad = grad;

      for (int iter = 0; iter < 20; ++iter)
      {
        const float midT = 0.5f * (leftT + rightT);
        float midVal, midDeriv;
        float3 midRst, midGrad;
        if (!EvaluateAlongRay(
              ops, ray, midT, isoValue, midRst, midVal, midGrad, midDeriv))
        {
          return false;
        }

        float candidateT = midT;
        if (fabsf(midDeriv) > 1e-8f)
        {
          const float newtonT = midT - midVal / midDeriv;
          if (newtonT > leftT && newtonT < rightT) candidateT = newtonT;
        }

        float candVal, candDeriv;
        float3 candRst, candGrad;
        if (!EvaluateAlongRay(
              ops, ray, candidateT, isoValue, candRst, candVal, candGrad, candDeriv))
        {
          candidateT = midT;
          candVal = midVal;
          candGrad = midGrad;
        }

        if (fabsf(candVal) < tolerance)
        {
          tHit = candidateT;
          rstHit = candRst;
          gradWorld = candGrad;
          return true;
        }

        if (leftVal * candVal <= 0.0f)
        {
          rightT = candidateT;
          rightVal = candVal;
          rightRst = candRst;
          rightGrad = candGrad;
        }
        else
        {
          leftT = candidateT;
          leftVal = candVal;
          leftRst = candRst;
          leftGrad = candGrad;
        }
      }

      // Fallback to bisection result
      tHit = 0.5f * (leftT + rightT);
      rstHit = leftRst;
      gradWorld = leftGrad;
      return true;
    }

    prevT = t;
    prevVal = value;
    prevRst = rst;
    prevGrad = grad;
    prevDeriv = deriv;
  }

  return false;
}
} // namespace

__host__ __device__ bool FindIsosurfaceOnRayHex(
  const HexElementData& elem,
  const Ray& ray,
  float tEnter,
  float tExit,
  float isoValue,
  float& tHit,
  float3& rstHit,
  float3& gradWorld,
  float tolerance)
{
  HexOps ops{elem};
  return FindIsosurfaceImpl(
    ops, ray, tEnter, tExit, isoValue, tHit, rstHit, gradWorld, tolerance);
}

__host__ __device__ bool FindIsosurfaceOnRayPrism(
  const PrismElementData& elem,
  const Ray& ray,
  float tEnter,
  float tExit,
  float isoValue,
  float& tHit,
  float3& rstHit,
  float3& gradWorld,
  float tolerance)
{
  PrismOps ops{elem};
  return FindIsosurfaceImpl(
    ops, ray, tEnter, tExit, isoValue, tHit, rstHit, gradWorld, tolerance);
}
} // namespace zidingyi
