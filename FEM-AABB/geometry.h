#ifndef GEOMETRY_H
#define GEOMETRY_H

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#include <cuda_runtime.h>
#include <vector_types.h>
#else
#ifndef __INTELLISENSE__
#include <cuda_runtime.h>
#include <vector_types.h>
#else
struct float3
{
    float x, y, z;
};
inline float3 make_float3(float x, float y, float z)
{
    return float3{x, y, z};
}
inline float3 make_float3(float v)
{
    return float3{v, v, v};
}
#endif
#endif
#include <cmath>

// --- 数学工具 ---

// float3的常用操作
__device__ __host__ inline float3 operator+(const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __host__ inline float3 operator-(const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __host__ inline float3 operator-(const float3& v) { return make_float3(-v.x, -v.y, -v.z); }
__device__ __host__ inline float3 operator*(const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ __host__ inline float3 operator*(float t, const float3& v) { return make_float3(t * v.x, t * v.y, t * v.z); }
__device__ __host__ inline float3 operator*(const float3& v, float t) { return t * v; }
__device__ __host__ inline float3 operator/(const float3& v, float t) { return v * (1.0f / t); }

__device__ __host__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__device__ __host__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__device__ __host__ inline float3& operator*=(float3& a, float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}
__device__ __host__ inline float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __host__ inline float3 cross(const float3& a, const float3& b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
__device__ __host__ inline float length_squared(const float3& v) { return dot(v, v); }
__device__ __host__ inline float length(const float3& v) { return sqrtf(length_squared(v)); }
__device__ __host__ inline float3 normalize(const float3& v) { return v / length(v); }


// --- 核心结构 ---

struct Ray {
    float3 origin;
    float3 direction;

    __device__ __host__ Ray() {}
    __device__ __host__ Ray(const float3& o, const float3& d) : origin(o), direction(d) {}

    __device__ __host__ float3 at(float t) const {
        return origin + t * direction;
    }
};

struct HitRecord {
    float t;
    float3 p;
    float3 normal;
    int material_id;
    bool front_face;

    __device__ __host__ inline void set_face_normal(const Ray& r, const float3& outward_normal) {
        front_face = dot(r.direction, outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

struct Material {
    float3 color;
    float3 emission; // 用于发光材质
    float roughness;
    float ior;      // Index of Refraction for dielectrics
    int type;       // 0: Lambertian, 1: Metal, 2: Dielectric

    Material() = default;
    Material(const float3& c, const float3& e, float rough, float index_of_refraction, int t) 
        : color(c), emission(e), roughness(rough), ior(index_of_refraction), type(t) {}
};

struct Sphere {
    float3 center;
    float radius;
    int material_id;

    Sphere() = default;
    Sphere(const float3& c, float r, int mid) : center(c), radius(r), material_id(mid) {}
};

struct Triangle {
    float3 v0, v1, v2;
    float3 normal;
    int material_id;

    Triangle() = default;
    Triangle(const float3& p0, const float3& p1, const float3& p2, int mid) 
        : v0(p0), v1(p1), v2(p2), material_id(mid) {
        normal = normalize(cross(v1 - v0, v2 - v0));
    }
};

struct AABB {
    float3 min_point;
    float3 max_point;

    __device__ __host__ AABB() {}
    __device__ __host__ AABB(const float3& minp, const float3& maxp) : min_point(minp), max_point(maxp) {}

    // 从球体创建AABB
    __host__ AABB(const Sphere& s) {
        float r = s.radius;
        min_point = make_float3(s.center.x - r, s.center.y - r, s.center.z - r);
        max_point = make_float3(s.center.x + r, s.center.y + r, s.center.z + r);
    }
    
    // 从三角形创建AABB
    __host__ AABB(const Triangle& t) {
        min_point = make_float3(fminf(t.v0.x, fminf(t.v1.x, t.v2.x)),
                                fminf(t.v0.y, fminf(t.v1.y, t.v2.y)),
                                fminf(t.v0.z, fminf(t.v1.z, t.v2.z)));
        max_point = make_float3(fmaxf(t.v0.x, fmaxf(t.v1.x, t.v2.x)),
                                fmaxf(t.v0.y, fmaxf(t.v1.y, t.v2.y)),
                                fmaxf(t.v0.z, fmaxf(t.v1.z, t.v2.z)));
    }

    // 合并两个AABB
    __host__ AABB union_aabb(const AABB& other) const {
        float3 new_min = make_float3(fminf(min_point.x, other.min_point.x),
                                     fminf(min_point.y, other.min_point.y),
                                     fminf(min_point.z, other.min_point.z));
        float3 new_max = make_float3(fmaxf(max_point.x, other.max_point.x),
                                     fmaxf(max_point.y, other.max_point.y),
                                     fmaxf(max_point.z, other.max_point.z));
        return AABB(new_min, new_max);
    }

    // AABB与光线相交测试 (Slab Test)
    __device__ __host__ bool intersect(const Ray& r, float& t_min, float& t_max) const {
        float3 invD = make_float3(1.0f / r.direction.x, 1.0f / r.direction.y, 1.0f / r.direction.z);
        float3 t0s = (min_point - r.origin) * invD;
        float3 t1s = (max_point - r.origin) * invD;
        
        if (invD.x < 0.0f) { float temp = t0s.x; t0s.x = t1s.x; t1s.x = temp; }
        if (invD.y < 0.0f) { float temp = t0s.y; t0s.y = t1s.y; t1s.y = temp; }
        if (invD.z < 0.0f) { float temp = t0s.z; t0s.z = t1s.z; t1s.z = temp; }

        t_min = fmaxf(t0s.x, fmaxf(t0s.y, t0s.z));
        t_max = fminf(t1s.x, fminf(t1s.y, t1s.z));

        return t_max > t_min && t_max > 0.0f;
    }
};

// --- BVH 节点类型（供设备与主机共同使用） ---
struct BVHNode {
    AABB bounds;
    int left_child_or_primitive; // 内部：左子索引；叶子：primitive_indices 起始
    int right_child_or_count;    // 内部：右子索引；叶子：图元数量
    bool is_leaf;
};
#endif
