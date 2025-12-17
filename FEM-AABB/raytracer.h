#ifndef RAYTRACER_H
#define RAYTRACER_H

#include "geometry.h"
#include "BVH.h"
#include "high_order.h"
#include <cuda_runtime.h>  
#include <vector>
#include <string>
#include <unordered_map>

// ==========================================
// 【新增】定义 Primitive (通用图元句柄)
// ==========================================
// 这是连接 main.cu 中注册逻辑和 BVH 构建器的关键
struct Primitive {
    PrimitiveType type; // 来自 high_order.h (Hex, Prism, Sphere 等)
    int index;          // 在对应数组中的索引
};

// 相机结构体
struct Camera {
    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;
    float lens_radius;

    Camera() {}
    Camera(float3 lookfrom, float3 lookat, float3 vup,
           float vfov, float aspect_ratio, float aperture, float focus_dist) {
        float theta = vfov * 3.14159265f / 180.0f;
        float h = tanf(theta / 2.0f);
        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;

        lens_radius = aperture / 2.0f;
    }
};

#ifndef RAYTRACER_NO_SCENE
// 场景结构体
struct Scene {
    struct FieldSlice
    {
        std::string name;
        bool is3D{true};
        uint3 modes3{make_uint3(0,0,0)};
        uint2 modes2{make_uint2(0,0)};
        size_t count{0};
        int offset{-1};
    };

    // 几何数据容器
    std::vector<Sphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<zidingyi::HexElementData> hexes;
    std::vector<zidingyi::PrismElementData> prisms;
    std::vector<zidingyi::QuadElementData> quads;
    std::vector<zidingyi::TriElementData> curved_tris;

    // Host-side coefficient buffers (for iso rendering)
    std::vector<float> hex_coefficients;
    std::vector<int> hex_coeff_offsets;
    std::vector<float> prism_coefficients;
    std::vector<int> prism_coeff_offsets;
    std::vector<float> quad_coefficients;
    std::vector<int> quad_coeff_offsets;
    std::vector<float> tri_coefficients;
    std::vector<int> tri_coeff_offsets;

    // Geometry (mapping) coefficient buffers
    std::vector<float> hex_geom_coefficients;
    std::vector<int> hex_geom_offsets;
    std::vector<float> prism_geom_coefficients;
    std::vector<int> prism_geom_offsets;
    std::vector<float> quad_geom_coefficients;
    std::vector<int> quad_geom_offsets;
    std::vector<float> tri_geom_coefficients;
    std::vector<int> tri_geom_offsets;

    // 多场切片信息（按元素存储）
    std::vector<std::vector<FieldSlice>> hex_field_slices;
    std::vector<std::vector<FieldSlice>> prism_field_slices;
    std::vector<std::vector<FieldSlice>> quad_field_slices;
    std::vector<std::vector<FieldSlice>> tri_field_slices;
    
    // 材质容器 (必须保留，否则渲染器不知道显示什么颜色)
    std::vector<Material> materials;

    // ==========================================
    // 【新增】通用对象列表
    // ==========================================
    // main.cu 会把所有加载的单元注册到这里，供 BVH 使用
    std::vector<Primitive> objects; 

    BVH bvh;
    Camera camera;
    int width, height;
    int samples_per_pixel;
    int max_depth;
    float iso_value{0.0f};
    bool use_iso{true};

    // 辅助添加函数
    void add_sphere(const Sphere& s) { spheres.push_back(s); }
    void add_triangle(const Triangle& t) { triangles.push_back(t); }
    void add_hex(const zidingyi::HexElementData& h,
                 const float* fieldCoeffs = nullptr,
                 uint3 fieldModes = make_uint3(0, 0, 0),
                 size_t field_count = 0,
                 const float* geomX = nullptr,
                 const float* geomY = nullptr,
                 const float* geomZ = nullptr,
                 uint3 geomModes = make_uint3(0, 0, 0),
                 size_t geom_count = 0,
                 const std::vector<FieldSlice>* slices = nullptr)
    {
        zidingyi::HexElementData elem = h;
        int offset = -1;
        if (fieldCoeffs && field_count > 0)
        {
            offset = static_cast<int>(hex_coefficients.size());
            hex_coeff_offsets.push_back(offset);
            hex_coefficients.insert(hex_coefficients.end(), fieldCoeffs, fieldCoeffs + field_count);
            if (fieldModes.x > 0) elem.fieldModes = fieldModes;
        }
        else
        {
            hex_coeff_offsets.push_back(-1);
        }

        elem.aabbMin = h.aabbMin;
        elem.aabbMax = h.aabbMax;

        int geom_offset = -1;
        if (geomX && geomY && geomZ && geom_count > 0)
        {
            geom_offset = static_cast<int>(hex_geom_coefficients.size());
            hex_geom_offsets.push_back(geom_offset);
            hex_geom_coefficients.insert(hex_geom_coefficients.end(), geomX, geomX + geom_count);
            hex_geom_coefficients.insert(hex_geom_coefficients.end(), geomY, geomY + geom_count);
            hex_geom_coefficients.insert(hex_geom_coefficients.end(), geomZ, geomZ + geom_count);
            if (geomModes.x > 0) elem.geomModes = geomModes;
        }
        else
        {
            hex_geom_offsets.push_back(-1);
        }

        elem.fieldCoefficients = nullptr; // patched before GPU upload
        elem.geomCoefficients[0] = elem.geomCoefficients[1] = elem.geomCoefficients[2] = nullptr;
        hexes.push_back(elem);
        if (slices) hex_field_slices.push_back(*slices);
        else hex_field_slices.emplace_back();
    }
    void add_prism(const zidingyi::PrismElementData& p,
                   const float* fieldCoeffs = nullptr,
                   uint3 fieldModes = make_uint3(0, 0, 0),
                   size_t field_count = 0,
                   const float* geomX = nullptr,
                   const float* geomY = nullptr,
                   const float* geomZ = nullptr,
                   uint3 geomModes = make_uint3(0, 0, 0),
                   size_t geom_count = 0,
                   const std::vector<FieldSlice>* slices = nullptr)
    {
        zidingyi::PrismElementData elem = p;
        int offset = -1;
        if (fieldCoeffs && field_count > 0)
        {
            offset = static_cast<int>(prism_coefficients.size());
            prism_coeff_offsets.push_back(offset);
            prism_coefficients.insert(prism_coefficients.end(), fieldCoeffs, fieldCoeffs + field_count);
            if (fieldModes.x > 0) elem.fieldModes = fieldModes;
        }
        else
        {
            prism_coeff_offsets.push_back(-1);
        }

        int geom_offset = -1;
        if (geomX && geomY && geomZ && geom_count > 0)
        {
            geom_offset = static_cast<int>(prism_geom_coefficients.size());
            prism_geom_offsets.push_back(geom_offset);
            prism_geom_coefficients.insert(prism_geom_coefficients.end(), geomX, geomX + geom_count);
            prism_geom_coefficients.insert(prism_geom_coefficients.end(), geomY, geomY + geom_count);
            prism_geom_coefficients.insert(prism_geom_coefficients.end(), geomZ, geomZ + geom_count);
            if (geomModes.x > 0) elem.geomModes = geomModes;
        }
        else
        {
            prism_geom_offsets.push_back(-1);
        }

        elem.fieldCoefficients = nullptr;
        elem.geomCoefficients[0] = elem.geomCoefficients[1] = elem.geomCoefficients[2] = nullptr;
        prisms.push_back(elem);
        if (slices) prism_field_slices.push_back(*slices);
        else prism_field_slices.emplace_back();
    }
    void add_quad(const zidingyi::QuadElementData& q,
                  const float* fieldCoeffs = nullptr,
                  uint2 fieldModes = make_uint2(0, 0),
                  size_t field_count = 0,
                  const float* geomX = nullptr,
                  const float* geomY = nullptr,
                  const float* geomZ = nullptr,
                  uint2 geomModes = make_uint2(0, 0),
                  size_t geom_count = 0,
                  const std::vector<FieldSlice>* slices = nullptr)
    {
        zidingyi::QuadElementData elem = q;
        int offset = -1;
        if (fieldCoeffs && field_count > 0)
        {
            offset = static_cast<int>(quad_coefficients.size());
            quad_coeff_offsets.push_back(offset);
            quad_coefficients.insert(quad_coefficients.end(), fieldCoeffs, fieldCoeffs + field_count);
            if (fieldModes.x > 0) elem.fieldModes = fieldModes;
        }
        else
        {
            quad_coeff_offsets.push_back(-1);
        }

        int geom_offset = -1;
        if (geomX && geomY && geomZ && geom_count > 0)
        {
            geom_offset = static_cast<int>(quad_geom_coefficients.size());
            quad_geom_offsets.push_back(geom_offset);
            quad_geom_coefficients.insert(quad_geom_coefficients.end(), geomX, geomX + geom_count);
            quad_geom_coefficients.insert(quad_geom_coefficients.end(), geomY, geomY + geom_count);
            quad_geom_coefficients.insert(quad_geom_coefficients.end(), geomZ, geomZ + geom_count);
            if (geomModes.x > 0) elem.geomModes = geomModes;
        }
        else
        {
            quad_geom_offsets.push_back(-1);
        }

        elem.fieldCoefficients = nullptr;
        elem.geomCoefficients[0] = elem.geomCoefficients[1] = elem.geomCoefficients[2] = nullptr;
        quads.push_back(elem);
        if (slices) quad_field_slices.push_back(*slices);
        else quad_field_slices.emplace_back();
    }
    void add_curved_tri(const zidingyi::TriElementData& tri,
                        const float* fieldCoeffs = nullptr,
                        uint2 fieldModes = make_uint2(0, 0),
                        size_t field_count = 0,
                        const float* geomX = nullptr,
                        const float* geomY = nullptr,
                        const float* geomZ = nullptr,
                        uint2 geomModes = make_uint2(0, 0),
                        size_t geom_count = 0,
                        const std::vector<FieldSlice>* slices = nullptr)
    {
        zidingyi::TriElementData elem = tri;
        int offset = -1;
        if (fieldCoeffs && field_count > 0)
        {
            offset = static_cast<int>(tri_coefficients.size());
            tri_coeff_offsets.push_back(offset);
            tri_coefficients.insert(tri_coefficients.end(), fieldCoeffs, fieldCoeffs + field_count);
            if (fieldModes.x > 0) elem.fieldModes = fieldModes;
        }
        else
        {
            tri_coeff_offsets.push_back(-1);
        }

        int geom_offset = -1;
        if (geomX && geomY && geomZ && geom_count > 0)
        {
            geom_offset = static_cast<int>(tri_geom_coefficients.size());
            tri_geom_offsets.push_back(geom_offset);
            tri_geom_coefficients.insert(tri_geom_coefficients.end(), geomX, geomX + geom_count);
            tri_geom_coefficients.insert(tri_geom_coefficients.end(), geomY, geomY + geom_count);
            tri_geom_coefficients.insert(tri_geom_coefficients.end(), geomZ, geomZ + geom_count);
            if (geomModes.x > 0) elem.geomModes = geomModes;
        }
        else
        {
            tri_geom_offsets.push_back(-1);
        }

        elem.fieldCoefficients = nullptr;
        elem.geomCoefficients[0] = elem.geomCoefficients[1] = elem.geomCoefficients[2] = nullptr;
        curved_tris.push_back(elem);
        if (slices) tri_field_slices.push_back(*slices);
        else tri_field_slices.emplace_back();
    }
    void add_material(const Material& m) { materials.push_back(m); }
    
    // 构建 BVH
    void build_bvh();
};
#endif

// 主渲染核函数声明
__global__ void render_kernel(
    uchar4* output,
    int width, int height, int samples_per_pixel, int max_depth,
    // --- Scene Data ---
    Sphere* d_spheres, int num_spheres,
    Triangle* d_triangles, int num_triangles,
    zidingyi::HexElementData* d_hexes, int num_hexes,
    zidingyi::PrismElementData* d_prisms, int num_prisms,
    zidingyi::QuadElementData* d_quads, int num_quads,
    zidingyi::TriElementData* d_curved_tris, int num_curved_tris,
    Material* d_materials, int num_materials,
    BVHNode* d_bvh_nodes, int num_bvh_nodes,
    int* d_bvh_primitives,
    // --- Camera Data ---
    Camera camera,
    float iso_value,
    // --- Random Seed ---
    unsigned int seed
);

// 图像保存函数
void save_image_ppm(const std::vector<uchar4>& image, int width, int height, const std::string& filename);

// Inline host-side scene helper
#ifndef RAYTRACER_NO_SCENE
inline void Scene::build_bvh() {
    // 注意：这里我们传入 objects，让 BVH 知道有哪些对象需要构建
    // 如果你的 BVH::build 签名还没改，可能需要去 BVH.h 里调整
    // 暂时保持你原有的调用方式，或者根据 objects 修改 BVH.cu
    // 假设 BVH.cu 内部会处理这些 vector
    bvh.build(spheres, triangles, hexes, prisms, quads, curved_tris);
}
#endif

#endif
