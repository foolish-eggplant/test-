#include "BVH.h"
#include <algorithm>
#include <stack>
#include <numeric> // For std::partition
#include <cfloat>  // For FLT_MAX
#include "high_order.h" 

// SAH 分箱所需的数据结构
struct SAHBin {
    AABB bounds;
    int primitive_count = 0;

    SAHBin() {
        // 初始化一个无效的包围盒
        bounds.min_point = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
        bounds.max_point = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    }

    void extend(const AABB& other) {
        bounds.min_point.x = fminf(bounds.min_point.x, other.min_point.x);
        bounds.min_point.y = fminf(bounds.min_point.y, other.min_point.y);
        bounds.min_point.z = fminf(bounds.min_point.z, other.min_point.z);
        bounds.max_point.x = fmaxf(bounds.max_point.x, other.max_point.x);
        bounds.max_point.y = fmaxf(bounds.max_point.y, other.max_point.y);
        bounds.max_point.z = fmaxf(bounds.max_point.z, other.max_point.z);
    }
};

// 辅助函数：计算AABB的表面积
inline float surface_area(const AABB& aabb) {
    float3 d = aabb.max_point - aabb.min_point;
    return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
}

int BVH::build_recursive(std::vector<int>& prim_indices, int start, int end,
                           const std::vector<AABB>& primitive_bounds) {
    int count = end - start;
    int node_idx = nodes.size();
    BVHNode node;

    // 计算当前范围内所有物体的总包围盒
    int first_prim_idx = prim_indices[start];
    int first_aabb_idx = DecodePrimitiveIndex(first_prim_idx);
    AABB total_bounds = primitive_bounds[first_aabb_idx];
    for (int i = start + 1; i < end; ++i) {
        int prim_idx = prim_indices[i];
        int aabb_idx = DecodePrimitiveIndex(prim_idx);
        total_bounds = total_bounds.union_aabb(primitive_bounds[aabb_idx]);
    }
    node.bounds = total_bounds;

    // --- SAH 核心逻辑开始 ---

    // 如果物体数量很少，直接创建叶子节点
    if (count <= 4) {
        node.is_leaf = true;
        node.left_child_or_primitive = start;
        node.right_child_or_count = count;
        nodes.push_back(node);
        return node_idx;
    }

    float parent_surface_area = surface_area(total_bounds);
    float best_cost = FLT_MAX;
    int best_axis = -1;
    int best_split_bin_idx = -1;

    const int NUM_BINS = 16;

    for (int axis = 0; axis < 3; ++axis) {
        SAHBin bins[NUM_BINS];
        float axis_min = total_bounds.min_point.x * (axis == 0) + total_bounds.min_point.y * (axis == 1) + total_bounds.min_point.z * (axis == 2);
        float axis_max = total_bounds.max_point.x * (axis == 0) + total_bounds.max_point.y * (axis == 1) + total_bounds.max_point.z * (axis == 2);

        if (axis_max - axis_min < 1e-6f) continue; // AABB 在该轴上是扁平的，无法分割

        // 1. 将所有图元放入箱子
        for (int i = start; i < end; ++i) {
            int prim_idx = prim_indices[i];
            int aabb_idx = DecodePrimitiveIndex(prim_idx);
            const AABB& prim_aabb = primitive_bounds[aabb_idx];
            
            // 计算图元中心点
            float center = (prim_aabb.min_point.x + prim_aabb.max_point.x) * (axis == 0) +
                           (prim_aabb.min_point.y + prim_aabb.max_point.y) * (axis == 1) +
                           (prim_aabb.min_point.z + prim_aabb.max_point.z) * (axis == 2);
            center *= 0.5f;

            int bin_idx = static_cast<int>(NUM_BINS * ((center - axis_min) / (axis_max - axis_min)));
            bin_idx = std::max(0, std::min(NUM_BINS - 1, bin_idx));

            bins[bin_idx].primitive_count++;
            bins[bin_idx].extend(prim_aabb);
        }

        // 2. 评估 N-1 个分割点的成本
        AABB left_aabb;
        int left_count = 0;
        for (int i = 0; i < NUM_BINS - 1; ++i) {
            if (bins[i].primitive_count > 0) {
                if (left_count == 0) left_aabb = bins[i].bounds;
                else left_aabb = left_aabb.union_aabb(bins[i].bounds);
                left_count += bins[i].primitive_count;
            }

            AABB right_aabb;
            int right_count = 0;
            for (int j = i + 1; j < NUM_BINS; ++j) {
                if (bins[j].primitive_count > 0) {
                    if (right_count == 0) right_aabb = bins[j].bounds;
                    else right_aabb = right_aabb.union_aabb(bins[j].bounds);
                    right_count += bins[j].primitive_count;
                }
            }

            if (left_count == 0 || right_count == 0) continue;

            float cost = (surface_area(left_aabb) * left_count + surface_area(right_aabb) * right_count) / parent_surface_area;
            
            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_split_bin_idx = i;
            }
        }
    }
    
    // 3. 决定是否分割
    float leaf_cost = count; // 不分割的成本就是与所有图元求交
    if (best_cost >= leaf_cost || best_axis == -1) {
        // 如果最佳分割成本不划算，或者没找到好的分割，则创建叶子节点
        node.is_leaf = true;
        node.left_child_or_primitive = start;
        node.right_child_or_count = count;
        nodes.push_back(node);
        return node_idx;
    }

    // 4. 执行分割
    // 使用 std::partition 将图元分为两组，比 std::sort 更高效
    float axis_min = total_bounds.min_point.x * (best_axis == 0) + total_bounds.min_point.y * (best_axis == 1) + total_bounds.min_point.z * (best_axis == 2);
    float axis_max = total_bounds.max_point.x * (best_axis == 0) + total_bounds.max_point.y * (best_axis == 1) + total_bounds.max_point.z * (best_axis == 2);
    float split_pos = axis_min + (best_split_bin_idx + 1) * (axis_max - axis_min) / NUM_BINS;

    auto* mid_ptr = std::partition(&prim_indices[start], &prim_indices[end],
        [&](int prim_idx) {
            int aabb_idx = DecodePrimitiveIndex(prim_idx);
            const AABB& prim_aabb = primitive_bounds[aabb_idx];
            float center = (prim_aabb.min_point.x + prim_aabb.max_point.x) * (best_axis == 0) +
                           (prim_aabb.min_point.y + prim_aabb.max_point.y) * (best_axis == 1) +
                           (prim_aabb.min_point.z + prim_aabb.max_point.z) * (best_axis == 2);
            return center * 0.5f < split_pos;
        });
    
    int mid = mid_ptr - &prim_indices[0];

    // 如果分割导致一边为空，则强制创建叶子节点
    if (mid == start || mid == end) {
        node.is_leaf = true;
        node.left_child_or_primitive = start;
        node.right_child_or_count = count;
        nodes.push_back(node);
        return node_idx;
    }

    // 创建内部节点并递归
    node.is_leaf = false;
    nodes.push_back(node); // 先占位，之后更新子节点索引

    int left_child = build_recursive(prim_indices, start, mid, primitive_bounds);
    int right_child = build_recursive(prim_indices, mid, end, primitive_bounds);

    nodes[node_idx].left_child_or_primitive = left_child;
    nodes[node_idx].right_child_or_count = right_child;

    return node_idx;
}

// 在CPU端构建BVH：创建图元索引与包围盒，并调用递归构建
void BVH::build(const std::vector<Sphere>& spheres,
               const std::vector<Triangle>& triangles,
               const std::vector<zidingyi::HexElementData>& hexes,
               const std::vector<zidingyi::PrismElementData>& prisms,
               const std::vector<zidingyi::QuadElementData>& quads,
               const std::vector<zidingyi::TriElementData>& curved_tris) {
    nodes.clear();
    primitive_indices.clear();

    const int num_spheres = static_cast<int>(spheres.size());
    const int num_triangles = static_cast<int>(triangles.size());
    const int num_hex = static_cast<int>(hexes.size());
    const int num_prism = static_cast<int>(prisms.size());
    const int num_quad = static_cast<int>(quads.size());
    const int num_curved_tri = static_cast<int>(curved_tris.size());

    // 编码规则：高 4bit 存储类型，低位存索引；primitive_bounds 与 prim_indices 同步压入，保持索引一致
    std::vector<int> prim_indices;
    std::vector<AABB> primitive_bounds;
    prim_indices.reserve(num_spheres + num_triangles + num_hex + num_prism + num_quad + num_curved_tri);
    primitive_bounds.reserve(prim_indices.size());

    auto push_prim = [&](int encoded_id, const AABB& box) {
        prim_indices.push_back(encoded_id);
        primitive_bounds.push_back(box);
    };

    for (int i = 0; i < num_spheres; ++i) {
        push_prim(EncodePrimitiveId(PrimitiveType::Sphere, i), AABB(spheres[i]));
    }
    for (int i = 0; i < num_triangles; ++i) {
        push_prim(EncodePrimitiveId(PrimitiveType::Triangle, i), AABB(triangles[i]));
    }
    for (int i = 0; i < num_hex; ++i) {
        float3 minC, maxC;
        minC = hexes[i].aabbMin;
        maxC = hexes[i].aabbMax;
        if (!(maxC.x >= minC.x && maxC.y >= minC.y && maxC.z >= minC.z)) {
            zidingyi::ComputeHexAabb(hexes[i], minC, maxC, 4);
        }
        push_prim(EncodePrimitiveId(PrimitiveType::Hex, i), AABB(minC, maxC));
    }
    for (int i = 0; i < num_prism; ++i) {
        float3 minC, maxC;
        zidingyi::ComputePrismAabb(prisms[i], minC, maxC, 4);
        push_prim(EncodePrimitiveId(PrimitiveType::Prism, i), AABB(minC, maxC));
    }
    for (int i = 0; i < num_quad; ++i) {
        float3 minC, maxC;
        zidingyi::ComputeQuadAabb(quads[i], minC, maxC, 4);
        push_prim(EncodePrimitiveId(PrimitiveType::Quad, i), AABB(minC, maxC));
    }
    for (int i = 0; i < num_curved_tri; ++i) {
        float3 minC, maxC;
        zidingyi::ComputeTriAabb(curved_tris[i], minC, maxC, 6);
        push_prim(EncodePrimitiveId(PrimitiveType::CurvedTriangle, i), AABB(minC, maxC));
    }

    // 若无图元，BVH为空
    if (prim_indices.empty()) {
        root = -1;
        return;
    }

    // 递归构建BVH，并保存最终的图元顺序
    root = build_recursive(prim_indices, 0, static_cast<int>(prim_indices.size()), primitive_bounds);
    primitive_indices = std::move(prim_indices);
}
