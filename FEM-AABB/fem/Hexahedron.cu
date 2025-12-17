#include "HighOrderCommon.cuh"
#include "HighOrderBasis.cuh"
#include <cuda_runtime.h>
// --- 复制开始 ---
#ifndef FLOAT3_OPERATORS_H
#define FLOAT3_OPERATORS_H

// 定义 float3 的减法 (a - b)
inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// 定义 float3 的加法 (a + b) - 为了防止以后报错，建议一起加上
inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

// 定义 float3 的乘法 (a * s)
inline __host__ __device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

// 定义 float3 的乘法 (s * a)
inline __host__ __device__ float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
#endif
// --- 复制结束 ---
namespace zidingyi
{
__device__ __forceinline__ float EvaluateHexAtReferencePoint(
  const float* coeffs, uint3 modes, const float3& p)
{
  float result = 0.0f;
  int idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float zk = ModifiedA(k, p.z);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float yj = ModifiedA(j, p.y);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float xi = ModifiedA(i, p.x);
        result += coeffs[idx++] * xi * yj * zk;
      }
    }
  }
  return result;
}

__device__ __forceinline__ float3 TransformReferenceToWorldLinearHex(
  const DeviceHighOrderScene& scene, int elementId, const float3& p)
{
  const float r = p.x;
  const float s = p.y;
  const float t = p.z;

  const float t1 = (1.0f - r) * (1.0f - s) * (1.0f - t);
  const float t2 = (1.0f + r) * (1.0f - s) * (1.0f - t);
  const float t3 = (1.0f + r) * (1.0f + s) * (1.0f - t);
  const float t4 = (1.0f - r) * (1.0f + s) * (1.0f - t);
  const float t5 = (1.0f - r) * (1.0f - s) * (1.0f + t);
  const float t6 = (1.0f + r) * (1.0f - s) * (1.0f + t);
  const float t7 = (1.0f + r) * (1.0f + s) * (1.0f + t);
  const float t8 = (1.0f - r) * (1.0f + s) * (1.0f + t);

  const float4 v0 = GetVertex(scene, elementId, 0);
  const float4 v1 = GetVertex(scene, elementId, 1);
  const float4 v2 = GetVertex(scene, elementId, 2);
  const float4 v3 = GetVertex(scene, elementId, 3);
  const float4 v4 = GetVertex(scene, elementId, 4);
  const float4 v5 = GetVertex(scene, elementId, 5);
  const float4 v6 = GetVertex(scene, elementId, 6);
  const float4 v7 = GetVertex(scene, elementId, 7);

  const float scale = 0.125f;
  const float x = scale * (t1 * v0.x + t2 * v1.x + t3 * v2.x + t4 * v3.x +
                           t5 * v4.x + t6 * v5.x + t7 * v6.x + t8 * v7.x);
  const float y = scale * (t1 * v0.y + t2 * v1.y + t3 * v2.y + t4 * v3.y +
                           t5 * v4.y + t6 * v5.y + t7 * v6.y + t8 * v7.y);
  const float z = scale * (t1 * v0.z + t2 * v1.z + t3 * v2.z + t4 * v3.z +
                           t5 * v4.z + t6 * v5.z + t7 * v6.z + t8 * v7.z);
  return make_float3(x, y, z);
}

__device__ __forceinline__ float3 TransformReferenceToWorldHex(
  const DeviceHighOrderScene& scene, int elementId, const float3& p)
{
  const float* geomX = GetCurvedGeom(scene, elementId, 0);
  if (!geomX) return TransformReferenceToWorldLinearHex(scene, elementId, p);
  const float* geomY = GetCurvedGeom(scene, elementId, 1);
  const float* geomZ = GetCurvedGeom(scene, elementId, 2);
  const uint3 modes = scene.curvedGeomNumModes[elementId];
  const int nm = modes.x * modes.y * modes.z;
  return make_float3(EvaluateHexAtReferencePoint(geomX, modes, p),
                     EvaluateHexAtReferencePoint(geomY, modes, p),
                     EvaluateHexAtReferencePoint(geomZ, modes, p));
}

__device__ __forceinline__ void CalculateJacobianLinearHex(
  const DeviceHighOrderScene& scene, int elementId, const float3& p, float J[9])
{
  const float r = p.x;
  const float s = p.y;
  const float t = p.z;

  const float t1 = 1.0f - s;
  const float t2 = 1.0f - t;
  const float t3 = t1 * t2;
  const float t6 = 1.0f + s;
  const float t7 = t6 * t2;
  const float t10 = 1.0f + t;
  const float t11 = t1 * t10;
  const float t14 = t6 * t10;
  const float t18 = 1.0f - r;
  const float t19 = t18 * t2;
  const float t21 = 1.0f + r;
  const float t22 = t21 * t2;
  const float t26 = t18 * t10;
  const float t28 = t21 * t10;
  const float t33 = t18 * t1;
  const float t35 = t21 * t1;
  const float t37 = t21 * t6;
  const float t39 = t18 * t6;

  const float4 v0 = GetVertex(scene, elementId, 0);
  const float4 v1 = GetVertex(scene, elementId, 1);
  const float4 v2 = GetVertex(scene, elementId, 2);
  const float4 v3 = GetVertex(scene, elementId, 3);
  const float4 v4 = GetVertex(scene, elementId, 4);
  const float4 v5 = GetVertex(scene, elementId, 5);
  const float4 v6 = GetVertex(scene, elementId, 6);
  const float4 v7 = GetVertex(scene, elementId, 7);

  const float s8 = 0.125f;

  J[0] = s8 * (-t3 * v0.x + t3 * v1.x + t7 * v2.x - t7 * v3.x - t11 * v4.x +
               t11 * v5.x + t14 * v6.x - t14 * v7.x);
  J[1] = s8 * (-t19 * v0.x - t22 * v1.x + t22 * v2.x + t19 * v3.x - t26 * v4.x -
               t28 * v5.x + t28 * v6.x + t26 * v7.x);
  J[2] = s8 * (-t33 * v0.x - t35 * v1.x - t37 * v2.x - t39 * v3.x + t33 * v4.x +
               t35 * v5.x + t37 * v6.x + t39 * v7.x);

  J[3] = s8 * (-t3 * v0.y + t3 * v1.y + t7 * v2.y - t7 * v3.y - t11 * v4.y +
               t11 * v5.y + t14 * v6.y - t14 * v7.y);
  J[4] = s8 * (-t19 * v0.y - t22 * v1.y + t22 * v2.y + t19 * v3.y - t26 * v4.y -
               t28 * v5.y + t28 * v6.y + t26 * v7.y);
  J[5] = s8 * (-t33 * v0.y - t35 * v1.y - t37 * v2.y - t39 * v3.y + t33 * v4.y +
               t35 * v5.y + t37 * v6.y + t39 * v7.y);

  J[6] = s8 * (-t3 * v0.z + t3 * v1.z + t7 * v2.z - t7 * v3.z - t11 * v4.z +
               t11 * v5.z + t14 * v6.z - t14 * v7.z);
  J[7] = s8 * (-t19 * v0.z - t22 * v1.z + t22 * v2.z + t19 * v3.z - t26 * v4.z -
               t28 * v5.z + t28 * v6.z + t26 * v7.z);
  J[8] = s8 * (-t33 * v0.z - t35 * v1.z - t37 * v2.z - t39 * v3.z + t33 * v4.z +
               t35 * v5.z + t37 * v6.z + t39 * v7.z);
}

__device__ __forceinline__ void CalculateInverseJacobianHex(
  const DeviceHighOrderScene& scene, int elementId, const float3& p, float J[9])
{
  CalculateJacobianLinearHex(scene, elementId, p, J);
  const float det = (-J[0] * J[4] * J[8] + J[0] * J[5] * J[7] + J[3] * J[1] * J[8] -
                     J[3] * J[2] * J[7] - J[6] * J[1] * J[5] + J[6] * J[2] * J[4]);
  const float invDet = 1.0f / det;
  const float a00 = (-J[4] * J[8] + J[5] * J[7]) * invDet;
  const float a01 = (J[1] * J[8] - J[2] * J[7]) * invDet;
  const float a02 = (J[1] * J[5] - J[2] * J[4]) * invDet;
  const float a10 = (J[3] * J[8] - J[5] * J[6]) * invDet;
  const float a11 = (-J[0] * J[8] + J[2] * J[6]) * invDet;
  const float a12 = (-J[0] * J[5] + J[2] * J[3]) * invDet;
  const float a20 = (-J[3] * J[7] + J[4] * J[6]) * invDet;
  const float a21 = (J[0] * J[7] - J[1] * J[6]) * invDet;
  const float a22 = (J[0] * J[4] - J[1] * J[3]) * invDet;

  J[0] = a00;
  J[1] = a01;
  J[2] = a02;
  J[3] = a10;
  J[4] = a11;
  J[5] = a12;
  J[6] = a20;
  J[7] = a21;
  J[8] = a22;
}

__device__ __forceinline__ bool TransformWorldToReferenceHex(
  const DeviceHighOrderScene& scene,
  int elementId,
  const float3& worldPoint,
  float3& referencePoint)
{
  referencePoint = make_float3(0.0f, 0.0f, 0.0f);
  constexpr int kMaxIterations = 100;
  constexpr float kTolerance = 1e-5f;
  for (int iter = 0; iter < kMaxIterations; ++iter)
  {
    const float3 mapped =
      TransformReferenceToWorldHex(scene, elementId, referencePoint);
    float3 diff = mapped - worldPoint;
    float J[9];
    CalculateInverseJacobianHex(scene, elementId, referencePoint, J);

    const float rAdjust =
      J[0] * diff.x + J[1] * diff.y + J[2] * diff.z;
    const float sAdjust =
      J[3] * diff.x + J[4] * diff.y + J[5] * diff.z;
    const float tAdjust =
      J[6] * diff.x + J[7] * diff.y + J[8] * diff.z;

    referencePoint.x -= rAdjust;
    referencePoint.y -= sAdjust;
    referencePoint.z -= tAdjust;

    if (fabsf(rAdjust) < kTolerance && fabsf(sAdjust) < kTolerance &&
        fabsf(tAdjust) < kTolerance)
    {
      return true;
    }
  }
  return false;
}
} // namespace zidingyi
