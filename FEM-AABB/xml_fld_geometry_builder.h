#pragma once

#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "xml_fld_geometry_parser.h"

bool BuildHexGeometry(const ElemHex& elem,
                      const std::vector<float3>& vertices,
                      const std::unordered_map<int, EdgeInfo>& edges,
                      const std::unordered_map<long long, int>& edgeLookup,
                      const std::unordered_map<int, CurvedEdge>& curvedEdges,
                      const std::unordered_map<int, CurvedFace>& curvedFaces,
                      const uint3& baseModes,
                      std::vector<float>& gx,
                      std::vector<float>& gy,
                      std::vector<float>& gz,
                      uint3& geomModes);

bool BuildQuadGeometry(const ElemQuad& elem,
                       const std::vector<float3>& vertices,
                       const std::unordered_map<long long, int>& edgeLookup,
                       const std::unordered_map<int, CurvedEdge>& curvedEdges,
                       const std::unordered_map<int, CurvedFace>& curvedFaces,
                       const uint2& baseModes,
                       std::vector<float>& gx,
                       std::vector<float>& gy,
                       std::vector<float>& gz,
                       uint2& geomModes);

bool BuildTriGeometry(const ElemTri& elem,
                      const std::vector<float3>& vertices,
                      const std::unordered_map<long long, int>& edgeLookup,
                      const std::unordered_map<int, EdgeInfo>& edges,
                      const std::unordered_map<int, CurvedEdge>& curvedEdges,
                      const uint2& baseModes,
                      std::vector<float>& gx,
                      std::vector<float>& gy,
                      std::vector<float>& gz,
                      uint2& geomModes);

bool BuildPrismGeometry(const ElemPrism& elem,
                        const std::vector<float3>& vertices,
                        const std::unordered_map<int, EdgeInfo>& edges,
                        const std::unordered_map<long long, int>& edgeLookup,
                        const std::unordered_map<int, CurvedEdge>& curvedEdges,
                        const std::unordered_map<int, CurvedFace>& curvedFaces,
                        const uint3& baseModes,
                        std::vector<float>& gx,
                        std::vector<float>& gy,
                        std::vector<float>& gz,
                        uint3& geomModes);
