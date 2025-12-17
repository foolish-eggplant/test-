#define RAYTRACER_NO_SCENE
#include "high_order.h"   // HexFaceParam / GetHexFaceParam
#include "raytracer.h"
#include <cuda_runtime.h>
#include "BVH.h"
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <cfloat>
#include <cstdio>

#include "fem/HexElement.cuh"   // 或 .h
#include "fem/PrismElement.cuh" // 或 .h
#include "fem/QuadElement.cuh"  // 或 .h
#include "fem/TriElement.cuh"   // 或 .h



// --- 设备端随机数工具 ---

__device__ float3 random_in_unit_disk(curandState* local_rand_state) {
    float3 p;
    do {
        p = 2.0f * make_float3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - make_float3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

__device__ float3 random_in_unit_sphere(curandState* local_rand_state) {
    float3 p;
    do {
        p = 2.0f * make_float3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - make_float3(1, 1, 1);
    } while (length_squared(p) >= 1.0f);
    return p;
}

// --- 设备端材质和光线交互 ---

__device__ float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
    float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
    return false;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

__device__ bool scatter(const Ray& r_in, const HitRecord& rec, const Material* materials,
                       float3& attenuation, Ray& scattered, curandState* local_rand_state) {
    const Material& mat = materials[rec.material_id];
    
    switch (mat.type) {
        case 0: { // Lambertian
            float3 scatter_direction = rec.normal + normalize(random_in_unit_sphere(local_rand_state));
            scattered = Ray(rec.p, scatter_direction);
            attenuation = mat.color;
            return true;
        }
        case 1: { // Metal
            float3 reflected = reflect(normalize(r_in.direction), rec.normal);
            scattered = Ray(rec.p, reflected + mat.roughness * random_in_unit_sphere(local_rand_state));
            attenuation = mat.color;
            return dot(scattered.direction, rec.normal) > 0;
        }
        case 2: { // Dielectric
            attenuation = make_float3(1.0f, 1.0f, 1.0f);
            float etai_over_etat = rec.front_face ? (1.0f / mat.ior) : mat.ior;
            
            float3 unit_direction = normalize(r_in.direction);
            float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
            float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

            if (etai_over_etat * sin_theta > 1.0f || schlick(cos_theta, etai_over_etat) > curand_uniform(local_rand_state)) {
                // Total internal reflection or reflectance
                float3 reflected = reflect(unit_direction, rec.normal);
                scattered = Ray(rec.p, reflected);
            } else {
                float3 refracted;
                refract(unit_direction, rec.normal, etai_over_etat, refracted);
                scattered = Ray(rec.p, refracted);
            }
            return true;
        }
    }
    return false;
}

// --- 设备端BVH相交测试 ---

__device__ bool bvh_intersect_device(const Ray& ray, HitRecord& rec,
                                     const BVHNode* d_nodes, const int* d_primitives,
                                     const Sphere* d_spheres, int num_spheres,
                                     const Triangle* d_triangles, int num_triangles,
                                     const zidingyi::HexElementData* d_hexes, int num_hexes,
                                     const zidingyi::PrismElementData* d_prisms, int num_prisms,
                                     const zidingyi::QuadElementData* d_quads, int num_quads,
                                     const zidingyi::TriElementData* d_curved_tris, int num_curved_tris,
                                     float iso_value) {
    if (d_nodes == nullptr) return false;

    rec.t = FLT_MAX;
    bool hit_anything = false;
    HitRecord temp_rec;

    int todo[64];
    int todo_idx = 0;
    todo[todo_idx++] = 0; // Start with root node

    while (todo_idx > 0) {
        int node_idx = todo[--todo_idx];
        const BVHNode& node = d_nodes[node_idx];

        float t_min_box, t_max_box;
        if (!node.bounds.intersect(ray, t_min_box, t_max_box) || t_min_box >= rec.t) {
            continue;
        }

        if (node.is_leaf) {
            for (int i = 0; i < node.right_child_or_count; ++i) {
                const int encoded = d_primitives[node.left_child_or_primitive + i];
                const PrimitiveType type = DecodePrimitiveType(encoded);
                const int prim_local_idx = DecodePrimitiveIndex(encoded);

                if (type == PrimitiveType::Sphere) {
                    if (prim_local_idx < num_spheres) {
                        const Sphere& sphere = d_spheres[prim_local_idx];
                        float3 oc = ray.origin - sphere.center;
                        float a = dot(ray.direction, ray.direction);
                        float b = 2.0f * dot(oc, ray.direction);
                        float c = dot(oc, oc) - sphere.radius * sphere.radius;
                        float discriminant = b * b - 4 * a * c;

                        if (discriminant > 0) {
                            float temp_t = (-b - sqrtf(discriminant)) / (2.0f * a);
                            if (temp_t > 0.001f && temp_t < rec.t) {
                                rec.t = temp_t;
                                rec.p = ray.at(temp_t);
                                rec.material_id = sphere.material_id;
                                rec.set_face_normal(ray, (rec.p - sphere.center) / sphere.radius);
                                hit_anything = true;
                            }
                        }
                    }
                } else if (type == PrimitiveType::Triangle) {
                    if (prim_local_idx < num_triangles) {
                        const Triangle& tri = d_triangles[prim_local_idx];
                        float3 edge1 = tri.v1 - tri.v0;
                        float3 edge2 = tri.v2 - tri.v0;
                        float3 h = cross(ray.direction, edge2);
                        float a = dot(edge1, h);

                        if (fabsf(a) > 1e-6f) {
                            float f = 1.0f / a;
                            float3 s = ray.origin - tri.v0;
                            float u = f * dot(s, h);
                            if (u >= 0.0f && u <= 1.0f) {
                                float3 q = cross(s, edge1);
                                float v = f * dot(ray.direction, q);
                                if (v >= 0.0f && u + v <= 1.0f) {
                                    float temp_t = f * dot(edge2, q);
                                    if (temp_t > 0.001f && temp_t < rec.t) {
                                        rec.t = temp_t;
                                        rec.p = ray.at(temp_t);
                                        rec.material_id = tri.material_id;
                                        rec.set_face_normal(ray, tri.normal);
                                        hit_anything = true;
                                    }
                                }
                            }
                        }
                    }
                } else if (type == PrimitiveType::Hex) {
                    if (prim_local_idx < num_hexes) {
                        const zidingyi::HexElementData& elem = d_hexes[prim_local_idx];
                        const bool hasHO = zidingyi::HasHighOrderGeom(elem);
                        if (prim_local_idx == 0 &&
                            threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
                            blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
                            printf("[GPU] Hex0 geomCoeffs=%p,%p,%p geomModes=(%u,%u,%u)\n",
                                   (const void*)elem.geomCoefficients[0],
                                   (const void*)elem.geomCoefficients[1],
                                   (const void*)elem.geomCoefficients[2],
                                   elem.geomModes.x, elem.geomModes.y, elem.geomModes.z);
                            printf("[GPU] Hex0 HasHighOrderGeom=%d\n", hasHO ? 1 : 0);
                        }
                        float3 minCorner, maxCorner;
                        zidingyi::ComputeHexAabb(elem, minCorner, maxCorner, 4);
                        float tEnter, tExit;
                        zidingyi::Ray horay = ToHighOrderRay(ray);
                        if (RayAabbIntersectHO(horay, minCorner, maxCorner, 0.001f, rec.t, tEnter, tExit)) {
                            float tHit;
                            float3 rstHit;
                            float3 gradWorld;
                            
                            bool iso_ok = (iso_value != 0.0f) && (elem.fieldCoefficients != nullptr) &&
                                          (elem.fieldModes.x > 0 && elem.fieldModes.y > 0 && elem.fieldModes.z > 0);
                            if (iso_ok && zidingyi::FindIsosurfaceOnRayHex(elem, horay, tEnter, tExit, iso_value, tHit, rstHit, gradWorld)) {
                                if (tHit > 0.001f && tHit < rec.t) {
                                    rec.t = tHit;
                                    rec.p = ray.at(tHit);
                                    rec.material_id = 0; // high-order material: use first by convention
                                    float3 n = normalize(gradWorld);
                                    if (dot(n, ray.direction) > 0.0f) n = -n;
                                    rec.set_face_normal(ray, n);
                                    hit_anything = true;
                                }
                            } else {
                                int hitFaceIdx = -1;
                                if (IntersectRayWithHex(elem, horay, 0.001f, rec.t, tHit, rstHit, hitFaceIdx)) {
                                    if (tHit > 0.001f && tHit < rec.t) {
                                        rec.t = tHit;
                                        rec.p = ray.at(tHit);
                                        float J[9];
                                        if (zidingyi::HasHighOrderGeom(elem))
                                            zidingyi::JacobianHighOrder(elem, rstHit, J);
                                        else
                                            zidingyi::JacobianTrilinear(elem, rstHit, J);
                                        float3 dPhi_dr = make_float3(J[0], J[3], J[6]);
                                        float3 dPhi_ds = make_float3(J[1], J[4], J[7]);
                                        float3 dPhi_dt = make_float3(J[2], J[5], J[8]);
                                        float3 n;
                                        if (fabsf(rstHit.x) > 0.9f) {
                                            // r = ±1, use (s, t) tangents
                                            n = cross(dPhi_ds, dPhi_dt);
                                        } else if (fabsf(rstHit.y) > 0.9f) {
                                            // s = ±1, use (r, t) tangents
                                            n = cross(dPhi_dr, dPhi_dt);
                                        } else {
                                            // t = ±1, use (r, s) tangents
                                            n = cross(dPhi_dr, dPhi_ds);
                                        }
                                        n = normalize(n);
                                        if (dot(n, ray.direction) > 0.0f) n = -n;
                                        rec.set_face_normal(ray, n);
                                        // 面 0 用蓝色材质（ID 1），其余保持灰色（ID 0）
                                        rec.material_id = (hitFaceIdx == 0) ? 1 : 0;
                                        hit_anything = true;
                                    }
                                }
                            }
                        }
                    }
                } else if (type == PrimitiveType::Prism) {
                    if (prim_local_idx < num_prisms) {
                        const zidingyi::PrismElementData& elem = d_prisms[prim_local_idx];
                        float3 minCorner, maxCorner;
                        zidingyi::ComputePrismAabb(elem, minCorner, maxCorner, 4);
                        float tEnter, tExit;
                        zidingyi::Ray horay = ToHighOrderRay(ray);
                        if (RayAabbIntersectHO(horay, minCorner, maxCorner, 0.001f, rec.t, tEnter, tExit)) {
                            float tHit;
                            float3 rstHit;
                            float3 gradWorld;
                            if (false && zidingyi::FindIsosurfaceOnRayPrism(elem, horay, tEnter, tExit, iso_value, tHit, rstHit, gradWorld)) {
                                if (tHit > 0.001f && tHit < rec.t) {
                                    rec.t = tHit;
                                    rec.p = ray.at(tHit);
                                    rec.material_id = 0;
                                    float3 n = normalize(gradWorld);
                                    if (dot(n, ray.direction) > 0.0f) n = -n;
                                    rec.set_face_normal(ray, n);
                                    hit_anything = true;
                                }
                            } else {
                                float isoTHit;
                                float3 rst;
                                if (zidingyi::IntersectRayWithPrism(elem, horay, 0.001f, rec.t, isoTHit, rst)) {
                                    if (isoTHit > 0.001f && isoTHit < rec.t) {
                                        rec.t = isoTHit;
                                        rec.p = ray.at(isoTHit);
                                        float J[9];
                                        zidingyi::JacobianPrism(elem, rst, J);
                                        float3 dPhi_dr = make_float3(J[0], J[3], J[6]);
                                        float3 dPhi_ds = make_float3(J[1], J[4], J[7]);
                                        float3 n = normalize(cross(dPhi_dr, dPhi_ds));
                                        if (dot(n, ray.direction) > 0.0f) n = -n;
                                        rec.set_face_normal(ray, n);
                                        rec.material_id = 0;
                                        hit_anything = true;
                                    }
                                }
                            }
                        }
                    }
                } else if (type == PrimitiveType::Quad) {
                    if (prim_local_idx < num_quads) {
                        const zidingyi::QuadElementData& elem = d_quads[prim_local_idx];
                        float3 minCorner, maxCorner;
                        zidingyi::ComputeQuadAabb(elem, minCorner, maxCorner, 4);
                        float tEnter, tExit;
                        zidingyi::Ray horay = ToHighOrderRay(ray);
                        if (RayAabbIntersectHO(horay, minCorner, maxCorner, 0.001f, rec.t, tEnter, tExit)) {
                            float tHit;
                            float2 rsHit;
                            if (zidingyi::IntersectRayWithQuad(elem, horay, 0.001f, rec.t, tHit, rsHit)) {
                                if (tHit > 0.001f && tHit < rec.t) {
                                    rec.t = tHit;
                                    rec.p = ray.at(tHit);
                                    float3 dPhi_dr, dPhi_ds;
                                    zidingyi::SurfaceJacobian(elem, rsHit, dPhi_dr, dPhi_ds);
                                    float3 n = normalize(cross(dPhi_dr, dPhi_ds));
                                    if (dot(n, ray.direction) > 0.0f) n = -n;
                                    rec.set_face_normal(ray, n);
                                    rec.material_id = 0;
                                    hit_anything = true;
                                }
                            }
                        }
                    }
                } else if (type == PrimitiveType::CurvedTriangle) {
                    if (prim_local_idx < num_curved_tris) {
                        const zidingyi::TriElementData& elem = d_curved_tris[prim_local_idx];
                        zidingyi::Ray horay = ToHighOrderRay(ray);
                        float3 minCorner, maxCorner;
                        zidingyi::ComputeTriAabb(elem, minCorner, maxCorner, 6);
                        float tEnter, tExit;
                        if (RayAabbIntersectHO(horay, minCorner, maxCorner, 0.001f, rec.t, tEnter, tExit)) {
                            float tHit;
                            float2 rsHit;
                            if (zidingyi::IntersectRayWithTriangle(elem, horay, 0.001f, rec.t, tHit, rsHit)) {
                                if (tHit > 0.001f && tHit < rec.t) {
                                    rec.t = tHit;
                                    rec.p = ray.at(tHit);
                                    float3 dPhi_dr, dPhi_ds;
                                    zidingyi::SurfaceJacobianTriangle(elem, rsHit, dPhi_dr, dPhi_ds);
                                    float3 n = normalize(cross(dPhi_dr, dPhi_ds));
                                    if (dot(n, ray.direction) > 0.0f) n = -n;
                                    rec.set_face_normal(ray, n);
                                    rec.material_id = 0;
                                    hit_anything = true;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Push children to stack. A simple optimization could be to push the closer child first.
            if (todo_idx < 62) { // Bounds check for safety
                todo[todo_idx++] = node.left_child_or_primitive;
                todo[todo_idx++] = node.right_child_or_count;
            }
        }
    }
    return hit_anything;
}

// --- 核心路径追踪逻辑 ---

__device__ float3 ray_color(Ray r,
                           Sphere* d_spheres, int num_spheres,
                           Triangle* d_triangles, int num_triangles,
                           zidingyi::HexElementData* d_hexes, int num_hexes,
                           zidingyi::PrismElementData* d_prisms, int num_prisms,
                           zidingyi::QuadElementData* d_quads, int num_quads,
                           zidingyi::TriElementData* d_curved_tris, int num_curved_tris,
                           Material* d_materials,
                           BVHNode* d_bvh_nodes, int* d_bvh_primitives,
                           int max_depth, float iso_value, curandState* local_rand_state) {
    float3 attenuation = make_float3(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < max_depth; ++i) {
        HitRecord rec;
        if (bvh_intersect_device(r, rec, d_bvh_nodes, d_bvh_primitives,
                                 d_spheres, num_spheres,
                                 d_triangles, num_triangles,
                                 d_hexes, num_hexes,
                                 d_prisms, num_prisms,
                                 d_quads, num_quads,
                                 d_curved_tris, num_curved_tris,
                                 iso_value)) {
            Ray scattered;
            float3 attenuation_scatter;
            float3 emission = d_materials[rec.material_id].emission;

            if (scatter(r, rec, d_materials, attenuation_scatter, scattered, local_rand_state)) {
                attenuation = attenuation * attenuation_scatter;
                r = scattered;
                if (dot(emission, emission) > 0.0f) { // If it's an emissive material that also scatters
                    return attenuation * emission;
                }
            } else {
                return attenuation * emission; // Hit an emissive material
            }
        } else {
            // Background / Sky
            float3 unit_direction = normalize(r.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            float3 sky_color = (1.0f - t) * make_float3(1.0f, 1.0f, 1.0f) + t * make_float3(0.5f, 0.7f, 1.0f);
            return attenuation * sky_color;
        }
    }
    return make_float3(0, 0, 0); // Exceeded max depth
}

// --- CUDA渲染主核函数 ---

__global__ void render_kernel(
    uchar4* output,
    int width, int height, int samples_per_pixel, int max_depth,
    Sphere* d_spheres, int num_spheres,
    Triangle* d_triangles, int num_triangles,
    zidingyi::HexElementData* d_hexes, int num_hexes,
    zidingyi::PrismElementData* d_prisms, int num_prisms,
    zidingyi::QuadElementData* d_quads, int num_quads,
    zidingyi::TriElementData* d_curved_tris, int num_curved_tris,
    Material* d_materials, int num_materials,
    BVHNode* d_bvh_nodes, int num_bvh_nodes,
    int* d_bvh_primitives,
    Camera camera,
    float iso_value,
    unsigned int seed) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    curandState local_rand_state;
    curand_init(seed + y * width + x, 0, 0, &local_rand_state);

    float3 color = make_float3(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        float u_rand = curand_uniform(&local_rand_state);
        float v_rand = curand_uniform(&local_rand_state);

        float u = (float(x) + u_rand) / float(width);
        float v = (float(y) + v_rand) / float(height);
        
        // Camera ray generation with depth of field
        float3 rd = camera.lens_radius * random_in_unit_disk(&local_rand_state);
        float3 offset = camera.u * rd.x + camera.v * rd.y;
        
        Ray r = Ray(camera.origin + offset, 
                    normalize(camera.lower_left_corner + u * camera.horizontal + v * camera.vertical - camera.origin - offset));

        color = color + ray_color(r,
                                  d_spheres, num_spheres,
                                  d_triangles, num_triangles,
                                  d_hexes, num_hexes,
                                  d_prisms, num_prisms,
                                  d_quads, num_quads,
                                  d_curved_tris, num_curved_tris,
                                  d_materials,
                                  d_bvh_nodes, d_bvh_primitives,
                                  max_depth, iso_value, &local_rand_state);
    }
    color = color / float(samples_per_pixel);

    // Gamma correction
    color = make_float3(sqrtf(color.x), sqrtf(color.y), sqrtf(color.z));

    // Clamp and convert to uchar4
    color.x = fminf(1.0f, color.x);
    color.y = fminf(1.0f, color.y);
    color.z = fminf(1.0f, color.z);
    
    output[y * width + x] = make_uchar4(
        (unsigned char)(255.99f * color.x),
        (unsigned char)(255.99f * color.y),
        (unsigned char)(255.99f * color.z),
        255
    );
}

// --- Host-side functions ---

void save_image_ppm(const std::vector<uchar4>& image, int width, int height, const std::string& filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.write((char*)&image[i].x, 1);
        file.write((char*)&image[i].y, 1);
        file.write((char*)&image[i].z, 1);
    }
    file.close();
}
