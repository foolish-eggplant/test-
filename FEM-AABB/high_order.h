#pragma once

#include <cfloat>
#include <cmath>
#include <algorithm>

#include "geometry.h"
#include "fem/HexElement.cuh"
#include "fem/PrismElement.cuh"
#include "fem/QuadElement.cuh"
#include "fem/TriElement.cuh"
#include "fem/RayIsosurface.cuh"
#include "fem/RayTypes.cuh"

// Primitive type tags packed into the upper bits of the encoded primitive id.
enum class PrimitiveType : int
{
    Sphere = 0,
    Triangle = 1,
    Hex = 2,
    Prism = 3,
    Quad = 4,
    CurvedTriangle = 5
};

inline __host__ __device__ int EncodePrimitiveId(PrimitiveType type, int index)
{
    return (static_cast<int>(type) << 28) | (index & 0x0FFFFFFF);
}

inline __host__ __device__ PrimitiveType DecodePrimitiveType(int encoded)
{
    return static_cast<PrimitiveType>((encoded >> 28) & 0xF);
}

inline __host__ __device__ int DecodePrimitiveIndex(int encoded)
{
    return encoded & 0x0FFFFFFF;
}

// Convert between app Ray and high-order Ray
inline __host__ __device__ zidingyi::Ray ToHighOrderRay(const Ray& r)
{
    zidingyi::Ray out;
    out.origin = r.origin;
    out.direction = r.direction;
    return out;
}

// AABB for hex; samples reference space when high-order geom exists.
inline __host__ __device__ void ComputeHexAabb(const zidingyi::HexElementData& elem,
                                               float3& minCorner,
                                               float3& maxCorner,
                                               int samplesPerAxis = 4)
{
    minCorner = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    maxCorner = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    if (!zidingyi::HasHighOrderGeom(elem) || samplesPerAxis <= 1)
    {
        for (int i = 0; i < 8; ++i)
        {
            const float3 v = elem.vertices[i];
            minCorner.x = fminf(minCorner.x, v.x);
            minCorner.y = fminf(minCorner.y, v.y);
            minCorner.z = fminf(minCorner.z, v.z);
            maxCorner.x = fmaxf(maxCorner.x, v.x);
            maxCorner.y = fmaxf(maxCorner.y, v.y);
            maxCorner.z = fmaxf(maxCorner.z, v.z);
        }
        // 轻微 padding 防止数值切掉边界
        const float pad = 1e-4f;
        minCorner.x -= pad; minCorner.y -= pad; minCorner.z -= pad;
        maxCorner.x += pad; maxCorner.y += pad; maxCorner.z += pad;
        return;
    }

    // 高阶：根据阶数自适应采样密度
  const int order = std::max(elem.geomModes.x, std::max(elem.geomModes.y, elem.geomModes.z));
  samplesPerAxis = std::max(samplesPerAxis, order + 2); // 至少 3 点

  for (int kz = 0; kz < samplesPerAxis; ++kz)
  {
        const float tz = -1.0f + 2.0f * static_cast<float>(kz) / static_cast<float>(samplesPerAxis - 1);
        for (int ky = 0; ky < samplesPerAxis; ++ky)
        {
            const float sy = -1.0f + 2.0f * static_cast<float>(ky) / static_cast<float>(samplesPerAxis - 1);
            for (int kx = 0; kx < samplesPerAxis; ++kx)
            {
                const float rx = -1.0f + 2.0f * static_cast<float>(kx) / static_cast<float>(samplesPerAxis - 1);
                const float3 rst = make_float3(rx, sy, tz);
                const float3 p = zidingyi::ReferenceToWorldHex(elem, rst);
                minCorner.x = fminf(minCorner.x, p.x);
                minCorner.y = fminf(minCorner.y, p.y);
                minCorner.z = fminf(minCorner.z, p.z);
                maxCorner.x = fmaxf(maxCorner.x, p.x);
                maxCorner.y = fmaxf(maxCorner.y, p.y);
                maxCorner.z = fmaxf(maxCorner.z, p.z);
            }
        }
    }

    // 按尺寸自适应 padding，阶数越高 padding 越大
    float3 size = make_float3(maxCorner.x - minCorner.x,
                              maxCorner.y - minCorner.y,
                              maxCorner.z - minCorner.z);
    const float padRatio = 0.15f;
    const float padX = fmaxf(size.x * padRatio, 1e-3f);
    const float padY = fmaxf(size.y * padRatio, 1e-3f);
    const float padZ = fmaxf(size.z * padRatio, 1e-3f);
    minCorner.x -= padX; minCorner.y -= padY; minCorner.z -= padZ;
    maxCorner.x += padX; maxCorner.y += padY; maxCorner.z += padZ;
}

inline __host__ __device__ bool RayAabbIntersectHO(const zidingyi::Ray& ray,
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
        const float origin = axis == 0 ? ray.origin.x : (axis == 1 ? ray.origin.y : ray.origin.z);
        const float direction = axis == 0 ? ray.direction.x : (axis == 1 ? ray.direction.y : ray.direction.z);
        const float minValue = axis == 0 ? minCorner.x : (axis == 1 ? minCorner.y : minCorner.z);
        const float maxValue = axis == 0 ? maxCorner.x : (axis == 1 ? maxCorner.y : maxCorner.z);

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

inline __host__ __device__ float ProjectPointToRayParameterHO(const zidingyi::Ray& ray, const float3& point)
{
    const float3 diff = make_float3(point.x - ray.origin.x,
                                    point.y - ray.origin.y,
                                    point.z - ray.origin.z);
    const float denom = dot(ray.direction, ray.direction);
    return denom > 0.0f ? dot(diff, ray.direction) / denom : 0.0f;
}

// Canonical face order helper: 0:z-,1:s-,2:r+,3:s+,4:r-,5:z+
struct HexFaceParam
{
    int   fixedAxis;
    float fixedValue;
    int   uAxis;
    int   vAxis;
};

inline __host__ __device__ HexFaceParam GetHexFaceParam(int faceIdx)
{
    switch (faceIdx)
    {
        case 0: return {2, -1.f, 0, 1}; // t=-1
        case 1: return {1, -1.f, 0, 2}; // s=-1
        case 2: return {0,  1.f, 1, 2}; // r=+1
        case 3: return {1,  1.f, 0, 2}; // s=+1
        case 4: return {0, -1.f, 1, 2}; // r=-1
        case 5: return {2,  1.f, 0, 1}; // t=+1
        default: return {2, -1.f, 0, 1};
    }
}

// Geometry-only ray/hex intersection (no field iso)
// 面级求交：faceIdx = 0..5
// 0: t = -1, 1: t = +1, 2: s = -1, 3: s = +1, 4: r = -1, 5: r = +1
inline __host__ __device__
bool IntersectRayWithFace(const zidingyi::HexElementData& elem,
                          const zidingyi::Ray&           ray,
                          int                            faceIdx,
                          float                          tMin,
                          float                          tMax,
                          float&                         tHit,
                          float3&                        rstHit,
                          float                          tolerance = 1e-4f,
                          int                            maxIter   = 50)
{
    const HexFaceParam fp = GetHexFaceParam(faceIdx);
    int   fixedAxis = fp.fixedAxis;   // 默认固定轴
    float fixedVal  = fp.fixedValue;
    int   uAxis     = fp.uAxis;       // 自由参数 (u,v) 分别对应 rst[uAxis], rst[vAxis]
    int   vAxis     = fp.vAxis;

    const float svals[3] = {-0.8f, 0.0f, 0.8f};
    float2 seeds[9];
    int seedCount = 0;
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i)
            seeds[seedCount++] = make_float2(svals[i], svals[j]);

    for (int si = 0; si < seedCount; ++si)
    {
        float rst[3] = {0.f, 0.f, 0.f};
        rst[fixedAxis] = fixedVal;
        rst[uAxis] = seeds[si].x;
        rst[vAxis] = seeds[si].y;

        float3 rstVec_init = make_float3(rst[0], rst[1], rst[2]);
        float3 X0  = zidingyi::ReferenceToWorldHex(elem, rstVec_init); 
        
        // 基于种子点的物理位置计算初始 t
        float tau0 = ProjectPointToRayParameterHO(ray, X0);
        float tau  = fminf(fmaxf(tau0, tMin), tMax);
        float prevRes = FLT_MAX;

        for (int iter = 0; iter < maxIter; ++iter)
        {
            float3 rstVec = make_float3(rst[0], rst[1], rst[2]);

            float3 X = zidingyi::ReferenceToWorldHex(elem, rstVec);

            float3 Y = make_float3(ray.origin.x  + tau * ray.direction.x,
                                   ray.origin.y  + tau * ray.direction.y,
                                   ray.origin.z  + tau * ray.direction.z);

            float3 R = make_float3(X.x - Y.x, X.y - Y.y, X.z - Y.z);

            float maxRes = fmaxf(fabsf(R.x), fmaxf(fabsf(R.y), fabsf(R.z)));
            if (maxRes < tolerance)
            {
                if (rst[uAxis] >= -1.001f && rst[uAxis] <= 1.001f &&
                    rst[vAxis] >= -1.001f && rst[vAxis] <= 1.001f &&
                    tau        >= tMin    && tau        <= tMax)
                {
                    tHit   = tau;
                    rstHit = rstVec;
                    return true;
                }
                break;
            }
            if (iter > 5 && maxRes > prevRes * 1.5f) break;
            prevRes = maxRes;

            float Jfull[9];
            if (zidingyi::HasHighOrderGeom(elem))
                zidingyi::JacobianHighOrder(elem, rstVec, Jfull);
            else
                zidingyi::JacobianTrilinear(elem, rstVec, Jfull);

            float3 dX_dr = make_float3(Jfull[0], Jfull[3], Jfull[6]);
            float3 dX_ds = make_float3(Jfull[1], Jfull[4], Jfull[7]);
            float3 dX_dt = make_float3(Jfull[2], Jfull[5], Jfull[8]);

            float3 col_u = (uAxis == 0) ? dX_dr : (uAxis == 1 ? dX_ds : dX_dt);
            float3 col_v = (vAxis == 0) ? dX_dr : (vAxis == 1 ? dX_ds : dX_dt);
            float3 col_tau = make_float3(-ray.direction.x,
                                         -ray.direction.y,
                                         -ray.direction.z);

            float J[9];
            // 行优先存储：每行是 (x,y,z)
            J[0] = col_u.x;   J[1] = col_v.x;   J[2] = col_tau.x;
            J[3] = col_u.y;   J[4] = col_v.y;   J[5] = col_tau.y;
            J[6] = col_u.z;   J[7] = col_v.z;   J[8] = col_tau.z;

            float invJ[9];
            zidingyi::InvertJacobian(J, invJ);

            const float3 negR = make_float3(-R.x, -R.y, -R.z);

            float du   = invJ[0]*negR.x + invJ[1]*negR.y + invJ[2]*negR.z;
            float dv   = invJ[3]*negR.x + invJ[4]*negR.y + invJ[5]*negR.z;
            float dTau = invJ[6]*negR.x + invJ[7]*negR.y + invJ[8]*negR.z;


            float maxD = fmaxf(fabsf(du), fmaxf(fabsf(dv), fabsf(dTau)));
            float scale = (maxD > 0.25f) ? (0.25f / maxD) : 1.0f;

            rst[uAxis] += du   * scale;
            rst[vAxis] += dv   * scale;
            tau        += dTau * scale;
            tau         = fminf(fmaxf(tau, tMin), tMax);

            // 防止跑飞
            rst[uAxis] = fminf(fmaxf(rst[uAxis], -1.5f), 1.5f);
            rst[vAxis] = fminf(fmaxf(rst[vAxis], -1.5f), 1.5f);
            rst[fixedAxis] = fixedVal;
        }
    }

    return false;
}

#ifndef HEX_DEBUG_TRI_FALLBACK
#define HEX_DEBUG_TRI_FALLBACK 0
#endif

// 简单三角化的调试求交（仅几何表面），用于验证高阶求交是否遗漏
inline __host__ __device__ bool IntersectRayWithHexTriangulated(const zidingyi::HexElementData& elem,
                                                                const zidingyi::Ray& ray,
                                                                float tMin,
                                                                float tMax,
                                                                float& tHit,
                                                                float3& rstHit,
                                                                int gridRes = 4)
{
    gridRes = std::min(6, std::max(gridRes, 3));
    float closestT = tMax;
    bool hit = false;

    auto tri_intersect = [&](const float3& a, const float3& b, const float3& c, float& tOut)->bool
    {
        const float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
        const float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
        const float3 pvec = cross(ray.direction, ac);
        const float det = dot(ab, pvec);
        if (fabsf(det) < 1e-8f) return false;
        const float invDet = 1.0f / det;
        const float3 tvec = make_float3(ray.origin.x - a.x, ray.origin.y - a.y, ray.origin.z - a.z);
        const float u = dot(tvec, pvec) * invDet;
        if (u < 0.0f || u > 1.0f) return false;
        const float3 qvec = cross(tvec, ab);
        const float v = dot(ray.direction, qvec) * invDet;
        if (v < 0.0f || u + v > 1.0f) return false;
        const float t = dot(ac, qvec) * invDet;
        if (t < tMin || t > tMax) return false;
        tOut = t;
        return true;
    };

    auto sample_face = [&](int faceIdx, float r, float s)->float3
    {
        float3 rst;
        switch (faceIdx)
        {
            // Face 0: Bottom (z=-1). r->r, s->s
            case 0: rst = make_float3(r, s, -1.0f); break;

            // Face 1: Front (y=-1). r->r, s->t
            case 1: rst = make_float3(r, -1.0f, s); break;

            // Face 2: Right (x=+1). r->s, s->t
            case 2: rst = make_float3(1.0f, r, s); break;

            // Face 3: Back (y=+1). r->r, s->t
            case 3: rst = make_float3(r, 1.0f, s); break;

            // Face 4: Left (x=-1). r->s, s->t
            case 4: rst = make_float3(-1.0f, r, s); break;

            // Face 5: Top (z=+1). r->r, s->s
            default: rst = make_float3(r, s, 1.0f); break;
        }
        return zidingyi::ReferenceToWorldHex(elem, rst);
    };

    for (int f = 0; f < 6; ++f)
    {
        for (int j = 0; j < gridRes - 1; ++j)
        {
            const float v0 = -1.0f + 2.0f * j / (gridRes - 1);
            const float v1 = -1.0f + 2.0f * (j + 1) / (gridRes - 1);
            for (int i = 0; i < gridRes - 1; ++i)
            {
                const float u0 = -1.0f + 2.0f * i / (gridRes - 1);
                const float u1 = -1.0f + 2.0f * (i + 1) / (gridRes - 1);

                float3 p00 = sample_face(f, u0, v0);
                float3 p10 = sample_face(f, u1, v0);
                float3 p01 = sample_face(f, u0, v1);
                float3 p11 = sample_face(f, u1, v1);

                float t;
                if (tri_intersect(p00, p10, p11, t) && t < closestT)
                {
                    closestT = t; hit = true;
                    rstHit = make_float3(u0, v0, 0); // 近似参考坐标记录
                }
                if (tri_intersect(p00, p11, p01, t) && t < closestT)
                {
                    closestT = t; hit = true;
                    rstHit = make_float3(u0, v0, 0);
                }
            }
        }
    }

    if (hit)
    {
        tHit = closestT;
        return true;
    }
    return false;
}

// Hex 主求交：用 AABB 粗裁剪，再遍历 6 个面取最近
inline __host__ __device__
bool IntersectRayWithHex(const zidingyi::HexElementData& elem,
                         const zidingyi::Ray&           ray,
                         float                          tMin,
                         float                          tMax,
                         float&                         tHit,
                         float3&                        rstHit,
                         int&                           hitFaceIdx,
                         int   /*maxIterations*/ = 60,
                         float tolerance          = 1e-4f)
{
  float3 minC, maxC;
    int samples = 0;
    if (zidingyi::HasHighOrderGeom(elem))
    {
        const int order = static_cast<int>(fmaxf(elem.geomModes.x, fmaxf(elem.geomModes.y, elem.geomModes.z)));
        samples = std::max(order + 2, 3);
    }
    ::ComputeHexAabb(elem, minC, maxC, samples);

    float tBoxEnter, tBoxExit;
    if (!RayAabbIntersectHO(ray, minC, maxC, tMin, tMax, tBoxEnter, tBoxExit))
        return false;

    bool   hitAny      = false;
    float  closestT    = tMax;
    float3 closestRst  = make_float3(0.f, 0.f, 0.f);
    int    closestFace = -1;

    for (int f = 0; f < 6; ++f)
    {
        float  tFace;
        float3 rstFace;
        if (IntersectRayWithFace(elem, ray, f,
                                 tMin, tMax,
                                 tFace, rstFace,
                                 tolerance))
        {
            if (tFace > tMin && tFace < closestT)
            {
                closestT    = tFace;
                closestRst  = rstFace;
                closestFace = f;
                hitAny      = true;
            }
        }
    }

    if (hitAny)
    {
        tHit       = closestT;
        rstHit     = closestRst;
        hitFaceIdx = closestFace;
        return true;
    }
    return false;
}

// Namespace alias for older call-sites.
namespace zidingyi
{
inline __host__ __device__ void ComputeHexAabb(const HexElementData& elem,
                                               float3& minCorner,
                                               float3& maxCorner,
                                               int samplesPerAxis = 4)
{
    ::ComputeHexAabb(elem, minCorner, maxCorner, samplesPerAxis);
}
} // namespace zidingyi
