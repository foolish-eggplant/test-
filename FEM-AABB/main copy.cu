#include "raytracer.h"
#include "scene_loader.h"
#include "xml_fld_loader.h"
#include <iostream>
#include <vector>
#include <time.h> // For random seed
#include <cfloat> // For FLT_MAX
#include <algorithm> // For min/max
#include <cctype>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t error = err; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main(int argc, char** argv) {
    // --- Image and Render Settings ---
    // 调试配置：减小分辨率/采样/递归深度，便于快速验证渲染
    const int width = 256;
    const int height = 144;
    const int samples_per_pixel = 1; // 采样数（调试）
    const int max_depth = 10;
    const float aspect_ratio = (float)width / (float)height;

    // --- Scene Setup ---
    Scene scene;
    scene.width = width;
    scene.height = height;
    scene.samples_per_pixel = samples_per_pixel;
    scene.max_depth = max_depth;
    scene.iso_value = 0.0f; // disable isosurface rendering, fallback to pure geometry

    // 添加默认材质 (Material ID 0) - 灰色漫反射
    scene.add_material(Material(make_float3(0.5f, 0.5f, 0.5f), make_float3(0,0,0), 0.0f, 0.0f, 0)); 

    // If a scene file is given, load it; otherwise try default Hex_CurvedFace.xml/.fld
    bool loaded_from_file = false;
    std::string scene_path = (argc >= 2) ? argv[1] : "Hex_CurvedFace.xml";
    std::string fld_override = (argc >= 3) ? argv[2] : "";

    std::cout << "Loading scene from file: " << scene_path << std::endl;
    auto to_lower_ext = [](const std::string& p) {
        auto dot = p.find_last_of('.');
        if (dot == std::string::npos) return std::string();
        std::string ext = p.substr(dot);
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
        return ext;
    };
    const std::string ext = to_lower_ext(scene_path);

    if (ext == ".xml" || ext == ".fld") {
        // Pair xml/fld: allow explicit second argument or infer sibling with same stem.
        std::string xml_path = scene_path;
        std::string fld_path = fld_override;
        if (ext == ".xml") {
            if (fld_path.empty() && scene_path.size() >= 4) {
                fld_path = scene_path.substr(0, scene_path.size() - 4) + ".fld";
            }
        } else { // .fld
            if (xml_path.size() >= 4) {
                xml_path = scene_path.substr(0, scene_path.size() - 4) + ".xml";
            }
            if (fld_path.empty()) fld_path = scene_path;
        }
        loaded_from_file = load_xml_fld(xml_path, fld_path, scene);
    } else {
        loaded_from_file = load_scene_file(scene_path, scene);
    }
    
    if (loaded_from_file) 
    {
        std::cout << "Registering loaded elements to scene objects..." << std::endl;
        
        for (size_t i = 0; i < scene.hexes.size(); ++i) scene.objects.push_back({PrimitiveType::Hex, static_cast<int>(i)});
        for (size_t i = 0; i < scene.prisms.size(); ++i) scene.objects.push_back({PrimitiveType::Prism, static_cast<int>(i)});
        for (size_t i = 0; i < scene.quads.size(); ++i) scene.objects.push_back({PrimitiveType::Quad, static_cast<int>(i)});
        for (size_t i = 0; i < scene.curved_tris.size(); ++i) scene.objects.push_back({PrimitiveType::CurvedTriangle, static_cast<int>(i)});
        for (size_t i = 0; i < scene.spheres.size(); ++i) scene.objects.push_back({PrimitiveType::Sphere, static_cast<int>(i)});
        for (size_t i = 0; i < scene.triangles.size(); ++i) scene.objects.push_back({PrimitiveType::Triangle, static_cast<int>(i)});
    } else {
        std::cerr << "[Error] Failed to load scene. Exiting." << std::endl;
        return 0;
    }

    // --- Camera Setup ---
    float3 lookfrom;
    float3 lookat;
    float dist_to_focus;
    float aperture = 0.0f; // 0.0 = 针孔相机(全清晰)

    if (loaded_from_file && !scene.objects.empty()) {
        // =========================================================
        // [自动相机] 计算模型包围盒，自动调整相机位置
        // 确保无论模型多大、多小、在哪里，都能被看到
        // =========================================================
        float3 min_bound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        float3 max_bound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        auto update_bounds = [&](const float3& v) {
            min_bound.x = fminf(min_bound.x, v.x);
            min_bound.y = fminf(min_bound.y, v.y);
            min_bound.z = fminf(min_bound.z, v.z);
            max_bound.x = fmaxf(max_bound.x, v.x);
            max_bound.y = fmaxf(max_bound.y, v.y);
            max_bound.z = fmaxf(max_bound.z, v.z);
        };

        // 遍历所有加载的单元顶点来计算包围盒
        for (const auto& hex : scene.hexes) { for(int i=0; i<8; ++i) update_bounds(hex.vertices[i]); }
        for (const auto& prism : scene.prisms) { for(int i=0; i<6; ++i) update_bounds(prism.vertices[i]); }
        for (const auto& quad : scene.quads) { for(int i=0; i<4; ++i) update_bounds(quad.vertices[i]); }
        for (const auto& tri : scene.curved_tris) {for (int i = 0; i < 3; ++i) update_bounds(tri.vertices[i]);}
        for (const auto& tri : scene.triangles) {update_bounds(tri.v0); update_bounds(tri.v1); update_bounds(tri.v2);}
        for (const auto& s : scene.spheres) {update_bounds(s.center - make_float3(s.radius, s.radius, s.radius));  // 球体的最小边界
            update_bounds(s.center + make_float3(s.radius, s.radius, s.radius));  // 球体的最大边界
}
        

        float3 center = (min_bound + max_bound) * 0.5f;
        float3 size = max_bound - min_bound;
        float max_dim = fmaxf(size.x, fmaxf(size.y, size.z));

        if (max_dim <= 0.0f) max_dim = 1.0f; // 防止除零

        std::cout << "Model Bounds: Min(" << min_bound.x << "," << min_bound.y << "," << min_bound.z << ") "
                  << "Max(" << max_bound.x << "," << max_bound.y << "," << max_bound.z << ")" << std::endl;

        // 设置相机看向中心
        lookat = center;
        // 3D 视角：对准包围盒中心，沿对角线偏移；若接近平面则正视面法线方向
        if (size.z < 1e-3f) {
            // 近似 2D：正视 z 轴方向，确保平面可见
            lookfrom = center + make_float3(0.0f, 0.0f, max_dim * 3.0f);
        } else {
            lookfrom = center + make_float3(max_dim * 1.2f, max_dim * 0.8f, max_dim * 1.6f);
        }
        dist_to_focus = length(lookfrom - lookat);
        aperture = 0.0f; // 禁用景深，避免近距离模糊
        
        std::cout << "Auto Camera: LookAt(" << lookat.x << "," << lookat.y << "," << lookat.z << ") "
                  << "Pos(" << lookfrom.x << "," << lookfrom.y << "," << lookfrom.z << ")" << std::endl;

    } else {
        std::cerr << "[Error] No geometry loaded. Exiting." << std::endl;
        return 0;
    }

    float3 vup = make_float3(0, 1, 0);
    scene.camera = Camera(lookfrom, lookat, vup, 20.0f, aspect_ratio, aperture, dist_to_focus);
    
    std::cout << "Scene setup complete. Total Objects: " << scene.objects.size() 
              << " (Hex:" << scene.hexes.size() 
              << " Prism:" << scene.prisms.size() 
              << " Quad:" << scene.quads.size() 
              << " Sphere:" << scene.spheres.size() << ")" << std::endl;
    
    // --- Build Acceleration Structure ---
    std::cout << "Building BVH..." << std::endl;
    scene.build_bvh();
    std::cout << "BVH built. Nodes: " << scene.bvh.nodes.size() << std::endl;

    // --- CUDA Memory Allocation and Transfer ---
    std::cout << "Allocating GPU memory..." << std::endl;
    uchar4* d_output;
    Sphere* d_spheres = nullptr;
    Triangle* d_triangles = nullptr;
    zidingyi::HexElementData* d_hexes = nullptr;
    zidingyi::PrismElementData* d_prisms = nullptr;
    zidingyi::QuadElementData* d_quads = nullptr;
    zidingyi::TriElementData* d_curved_tris = nullptr;
    float* d_hex_coeffs = nullptr;
    float* d_prism_coeffs = nullptr;
    float* d_quad_coeffs = nullptr;
    float* d_tri_coeffs = nullptr;
    float* d_hex_geom = nullptr;
    float* d_prism_geom = nullptr;
    float* d_quad_geom = nullptr;
    float* d_tri_geom = nullptr;
    Material* d_materials = nullptr;
    BVHNode* d_bvh_nodes = nullptr;
    int* d_bvh_primitives = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_output, width * height * sizeof(uchar4)));

    // Patch coefficient pointers and upload modal data (if available)
    auto hex_coeff_count = [](uint3 m) -> size_t {
        return static_cast<size_t>(m.x) * m.y * m.z;
    };
    auto quad_coeff_count = [](uint2 m) -> size_t {
        return static_cast<size_t>(m.x) * m.y;
    };
    auto tri_coeff_count = [](uint2 m) -> size_t {
        size_t count = 0;
        for (unsigned int i = 0; i < m.x; ++i) {
            unsigned int maxJ = (m.y > i) ? (m.y - i) : 0;
            count += maxJ;
        }
        return count;
    };
    auto prism_coeff_count = [](uint3 m) -> size_t {
        size_t count = 0;
        for (unsigned int i = 0; i < m.x; ++i) {
            unsigned int maxK = (m.z > i) ? (m.z - i) : 0;
            count += static_cast<size_t>(m.y) * maxK;
        }
        return count;
    };

    if (!scene.hex_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_hex_coeffs, scene.hex_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_hex_coeffs, scene.hex_coefficients.data(),
                              scene.hex_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.hexes.size(); ++i) {
            int offset = (i < scene.hex_coeff_offsets.size()) ? scene.hex_coeff_offsets[i] : -1;
            scene.hexes[i].fieldCoefficients = (offset >= 0) ? (d_hex_coeffs + offset) : nullptr;
        }
    }
    if (!scene.prism_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_prism_coeffs, scene.prism_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_prism_coeffs, scene.prism_coefficients.data(),
                              scene.prism_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.prisms.size(); ++i) {
            int offset = (i < scene.prism_coeff_offsets.size()) ? scene.prism_coeff_offsets[i] : -1;
            scene.prisms[i].fieldCoefficients = (offset >= 0) ? (d_prism_coeffs + offset) : nullptr;
        }
    }
    if (!scene.quad_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_quad_coeffs, scene.quad_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_quad_coeffs, scene.quad_coefficients.data(),
                              scene.quad_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.quads.size(); ++i) {
            int offset = (i < scene.quad_coeff_offsets.size()) ? scene.quad_coeff_offsets[i] : -1;
            scene.quads[i].fieldCoefficients = (offset >= 0) ? (d_quad_coeffs + offset) : nullptr;
        }
    }
    if (!scene.tri_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_tri_coeffs, scene.tri_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tri_coeffs, scene.tri_coefficients.data(),
                              scene.tri_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.curved_tris.size(); ++i) {
            int offset = (i < scene.tri_coeff_offsets.size()) ? scene.tri_coeff_offsets[i] : -1;
            scene.curved_tris[i].fieldCoefficients = (offset >= 0) ? (d_tri_coeffs + offset) : nullptr;
        }
    }

    // Upload geometry coefficients for high-order mappings
    if (!scene.hex_geom_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_hex_geom, scene.hex_geom_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_hex_geom, scene.hex_geom_coefficients.data(),
                              scene.hex_geom_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.hexes.size(); ++i) {
            int offset = (i < scene.hex_geom_offsets.size()) ? scene.hex_geom_offsets[i] : -1;
            const size_t count = hex_coeff_count(scene.hexes[i].geomModes);
            if (offset >= 0 && count > 0) {
                scene.hexes[i].geomCoefficients[0] = d_hex_geom + offset;
                scene.hexes[i].geomCoefficients[1] = d_hex_geom + offset + count;
                scene.hexes[i].geomCoefficients[2] = d_hex_geom + offset + 2 * count;
            } else {
                scene.hexes[i].geomCoefficients[0] = scene.hexes[i].geomCoefficients[1] = scene.hexes[i].geomCoefficients[2] = nullptr;
            }
        }
    }
    if (!scene.prism_geom_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_prism_geom, scene.prism_geom_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_prism_geom, scene.prism_geom_coefficients.data(),
                              scene.prism_geom_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.prisms.size(); ++i) {
            int offset = (i < scene.prism_geom_offsets.size()) ? scene.prism_geom_offsets[i] : -1;
            const size_t count = prism_coeff_count(scene.prisms[i].geomModes);
            if (offset >= 0 && count > 0) {
                scene.prisms[i].geomCoefficients[0] = d_prism_geom + offset;
                scene.prisms[i].geomCoefficients[1] = d_prism_geom + offset + count;
                scene.prisms[i].geomCoefficients[2] = d_prism_geom + offset + 2 * count;
            } else {
                scene.prisms[i].geomCoefficients[0] = scene.prisms[i].geomCoefficients[1] = scene.prisms[i].geomCoefficients[2] = nullptr;
            }
        }
    }
    if (!scene.quad_geom_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_quad_geom, scene.quad_geom_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_quad_geom, scene.quad_geom_coefficients.data(),
                              scene.quad_geom_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.quads.size(); ++i) {
            int offset = (i < scene.quad_geom_offsets.size()) ? scene.quad_geom_offsets[i] : -1;
            const size_t count = quad_coeff_count(scene.quads[i].geomModes);
            if (offset >= 0 && count > 0) {
                scene.quads[i].geomCoefficients[0] = d_quad_geom + offset;
                scene.quads[i].geomCoefficients[1] = d_quad_geom + offset + count;
                scene.quads[i].geomCoefficients[2] = d_quad_geom + offset + 2 * count;
            } else {
                scene.quads[i].geomCoefficients[0] = scene.quads[i].geomCoefficients[1] = scene.quads[i].geomCoefficients[2] = nullptr;
            }
        }
    }
    if (!scene.tri_geom_coefficients.empty()) {
        CUDA_CHECK(cudaMalloc(&d_tri_geom, scene.tri_geom_coefficients.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_tri_geom, scene.tri_geom_coefficients.data(),
                              scene.tri_geom_coefficients.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        for (size_t i = 0; i < scene.curved_tris.size(); ++i) {
            int offset = (i < scene.tri_geom_offsets.size()) ? scene.tri_geom_offsets[i] : -1;
            const size_t count = tri_coeff_count(scene.curved_tris[i].geomModes);
            if (offset >= 0 && count > 0) {
                scene.curved_tris[i].geomCoefficients[0] = d_tri_geom + offset;
                scene.curved_tris[i].geomCoefficients[1] = d_tri_geom + offset + count;
                scene.curved_tris[i].geomCoefficients[2] = d_tri_geom + offset + 2 * count;
            } else {
                scene.curved_tris[i].geomCoefficients[0] = scene.curved_tris[i].geomCoefficients[1] = scene.curved_tris[i].geomCoefficients[2] = nullptr;
            }
        }
    }
    
    // 拷贝各种几何数据到 GPU
    if (!scene.spheres.empty()) {
        CUDA_CHECK(cudaMalloc(&d_spheres, scene.spheres.size() * sizeof(Sphere)));
        CUDA_CHECK(cudaMemcpy(d_spheres, scene.spheres.data(), scene.spheres.size() * sizeof(Sphere), cudaMemcpyHostToDevice));
    }
    if (!scene.triangles.empty()) {
        CUDA_CHECK(cudaMalloc(&d_triangles, scene.triangles.size() * sizeof(Triangle)));
        CUDA_CHECK(cudaMemcpy(d_triangles, scene.triangles.data(), scene.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice));
    }
    if (!scene.hexes.empty()) {
        CUDA_CHECK(cudaMalloc(&d_hexes, scene.hexes.size() * sizeof(zidingyi::HexElementData)));
        CUDA_CHECK(cudaMemcpy(d_hexes, scene.hexes.data(), scene.hexes.size() * sizeof(zidingyi::HexElementData), cudaMemcpyHostToDevice));
    }
    if (!scene.prisms.empty()) {
        CUDA_CHECK(cudaMalloc(&d_prisms, scene.prisms.size() * sizeof(zidingyi::PrismElementData)));
        CUDA_CHECK(cudaMemcpy(d_prisms, scene.prisms.data(), scene.prisms.size() * sizeof(zidingyi::PrismElementData), cudaMemcpyHostToDevice));
    }
    if (!scene.quads.empty()) {
        CUDA_CHECK(cudaMalloc(&d_quads, scene.quads.size() * sizeof(zidingyi::QuadElementData)));
        CUDA_CHECK(cudaMemcpy(d_quads, scene.quads.data(), scene.quads.size() * sizeof(zidingyi::QuadElementData), cudaMemcpyHostToDevice));
    }
    if (!scene.curved_tris.empty()) {
        CUDA_CHECK(cudaMalloc(&d_curved_tris, scene.curved_tris.size() * sizeof(zidingyi::TriElementData)));
        CUDA_CHECK(cudaMemcpy(d_curved_tris, scene.curved_tris.data(), scene.curved_tris.size() * sizeof(zidingyi::TriElementData), cudaMemcpyHostToDevice));
    }
    if (!scene.materials.empty()) {
        CUDA_CHECK(cudaMalloc(&d_materials, scene.materials.size() * sizeof(Material)));
        CUDA_CHECK(cudaMemcpy(d_materials, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice));
    }
    // 拷贝 BVH 树结构
    if (!scene.bvh.nodes.empty()) {
        CUDA_CHECK(cudaMalloc(&d_bvh_nodes, scene.bvh.nodes.size() * sizeof(BVHNode)));
        CUDA_CHECK(cudaMemcpy(d_bvh_nodes, scene.bvh.nodes.data(), scene.bvh.nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
    }
    // 拷贝 BVH 叶子节点索引
    if (!scene.bvh.primitive_indices.empty()) {
        CUDA_CHECK(cudaMalloc(&d_bvh_primitives, scene.bvh.primitive_indices.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_bvh_primitives, scene.bvh.primitive_indices.data(), scene.bvh.primitive_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // --- Launch CUDA Kernel ---
    std::cout << "Rendering..." << std::endl;
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    render_kernel<<<gridSize, blockSize>>>(
        d_output, width, height, samples_per_pixel, max_depth,
        d_spheres, scene.spheres.size(),
        d_triangles, scene.triangles.size(),
        d_hexes, scene.hexes.size(),
        d_prisms, scene.prisms.size(),
        d_quads, scene.quads.size(),
        d_curved_tris, scene.curved_tris.size(),
        d_materials, scene.materials.size(),
        d_bvh_nodes, scene.bvh.nodes.size(),
        d_bvh_primitives,
        scene.camera,
        scene.iso_value,
        time(0) // Seed for cuRAND ·
    );
    
    // --- Error Checking and Synchronization ---
    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel launch
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to finish
    std::cout << "Render complete." << std::endl;
    
    // --- Copy Result Back to Host ---
    std::vector<uchar4> h_output(width * height);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
    
    // --- Save Image and Cleanup ---
    save_image_ppm(h_output, width, height, "output.ppm");
    std::cout << "Image saved as output.ppm" << std::endl;
    
    cudaFree(d_output);
    if (d_spheres) cudaFree(d_spheres);
    if (d_triangles) cudaFree(d_triangles);
    if (d_hex_coeffs) cudaFree(d_hex_coeffs);
    if (d_hex_geom) cudaFree(d_hex_geom);
    if (d_hexes) cudaFree(d_hexes);
    if (d_prism_coeffs) cudaFree(d_prism_coeffs);
    if (d_prism_geom) cudaFree(d_prism_geom);
    if (d_prisms) cudaFree(d_prisms);
    if (d_quad_coeffs) cudaFree(d_quad_coeffs);
    if (d_quad_geom) cudaFree(d_quad_geom);
    if (d_quads) cudaFree(d_quads);
    if (d_tri_coeffs) cudaFree(d_tri_coeffs);
    if (d_tri_geom) cudaFree(d_tri_geom);
    if (d_curved_tris) cudaFree(d_curved_tris);
    if (d_materials) cudaFree(d_materials);
    if (d_bvh_nodes) cudaFree(d_bvh_nodes);
    if (d_bvh_primitives) cudaFree(d_bvh_primitives);
    
    system("pause"); // 可选：如果需要保留窗口
    return 0;
}
