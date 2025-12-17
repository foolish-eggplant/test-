#ifndef BVH_H
#define BVH_H

#include "geometry.h"
#include "high_order.h"
#include <vector>

// BVHNode 现由 geometry.h 提供

class BVH {
public:
    int root = -1;
    std::vector<BVHNode> nodes;
    std::vector<int> primitive_indices;

    // 在CPU端构建BVH
    void build(const std::vector<Sphere>& spheres,
               const std::vector<Triangle>& triangles,
               const std::vector<zidingyi::HexElementData>& hexes,
               const std::vector<zidingyi::PrismElementData>& prisms,
               const std::vector<zidingyi::QuadElementData>& quads,
               const std::vector<zidingyi::TriElementData>& curved_tris);

private:
    int build_recursive(std::vector<int>& prim_indices, int start, int end, 
                        const std::vector<AABB>& primitive_bounds);
};
#endif
