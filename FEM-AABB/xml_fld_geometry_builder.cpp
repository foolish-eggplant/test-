#include "xml_fld_geometry_builder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <functional>
#include <cstdio>

#include "xml_fld_math.h"

namespace
{
float3 FetchVertex(const std::vector<float3>& vertices, int vid)
{
  if (vid >= 0 && vid < static_cast<int>(vertices.size())) return vertices[vid];
  return make_float3(0, 0, 0);
}

struct PrismNodeKeyHasher
{
  std::size_t operator()(const uint64_t& k) const noexcept { return std::hash<uint64_t>{}(k); }
};

inline uint64_t PrismNodeKey(unsigned int i, unsigned int j, unsigned int k)
{
  return (static_cast<uint64_t>(i) << 42) | (static_cast<uint64_t>(j) << 21) | static_cast<uint64_t>(k);
}

inline float EvaluatePrismBasis(unsigned int i, unsigned int j, unsigned int k, const float3& rst)
{
  const float ri = zidingyi::ModifiedA(i, rst.x);
  const float sj = zidingyi::ModifiedA(j, rst.y);
  const float tk = zidingyi::ModifiedB(i, k, rst.z);
  return ri * sj * tk;
}

inline float3 ReferenceToWorldPrismLinear(const std::array<float3, 6>& v, const float3& rst)
{
  const float r = rst.x;
  const float s = rst.y;
  const float t = rst.z;
  const float t1 = (1.0f - r) * (1.0f - s) * (1.0f - t);
  const float t2 = (1.0f + r) * (1.0f - s) * (1.0f - t);
  const float t3 = (1.0f + r) * (1.0f + s) * (1.0f - t);
  const float t4 = (1.0f - r) * (1.0f + s) * (1.0f - t);
  const float t5 = (1.0f - s) * (1.0f + t);
  const float t6 = (1.0f + s) * (1.0f + t);
  const float scaleQuad = 0.125f;
  const float scaleTop = 0.25f;
  float3 result = make_float3(0, 0, 0);
  auto addw = [&](int idx, float w)
  {
    result.x += w * v[idx].x;
    result.y += w * v[idx].y;
    result.z += w * v[idx].z;
  };
  addw(0, scaleQuad * t1);
  addw(1, scaleQuad * t2);
  addw(2, scaleQuad * t3);
  addw(3, scaleQuad * t4);
  addw(4, scaleTop * t5);
  addw(5, scaleTop * t6);
  return result;
}
} // namespace

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
                      uint3& geomModes)
{
  const int hexEdgeVerts[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {0, 4}, {1, 5}, {2, 6}, {3, 7},
      {4, 5}, {5, 6}, {6, 7}, {7, 4}};

  auto face_corners = [&](int f) -> std::array<float3, 4>
  {
    switch (f)
    {
      case 0: return {FetchVertex(vertices, elem.verts[0]), FetchVertex(vertices, elem.verts[1]), FetchVertex(vertices, elem.verts[2]), FetchVertex(vertices, elem.verts[3])}; // z-
      case 1: return {FetchVertex(vertices, elem.verts[0]), FetchVertex(vertices, elem.verts[1]), FetchVertex(vertices, elem.verts[5]), FetchVertex(vertices, elem.verts[4])}; // s-
      case 2: return {FetchVertex(vertices, elem.verts[1]), FetchVertex(vertices, elem.verts[2]), FetchVertex(vertices, elem.verts[6]), FetchVertex(vertices, elem.verts[5])}; // r+
      case 3: return {FetchVertex(vertices, elem.verts[3]), FetchVertex(vertices, elem.verts[2]), FetchVertex(vertices, elem.verts[6]), FetchVertex(vertices, elem.verts[7])}; // s+
      case 4: return {FetchVertex(vertices, elem.verts[0]), FetchVertex(vertices, elem.verts[3]), FetchVertex(vertices, elem.verts[7]), FetchVertex(vertices, elem.verts[4])}; // r-
      case 5: return {FetchVertex(vertices, elem.verts[4]), FetchVertex(vertices, elem.verts[5]), FetchVertex(vertices, elem.verts[6]), FetchVertex(vertices, elem.verts[7])}; // z+
      default: return {make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0)};
    }
  };

  auto blend_face = [&](int lf,
                        const std::vector<float3>& faceGrid,
                        unsigned int dimU,
                        unsigned int dimV,
                        std::vector<float3>& nodal,
                        const std::vector<float>& rNodes,
                        const std::vector<float>& sNodes,
                        const std::vector<float>& tNodes,
                        int nx,
                        int ny,
                        int nz,
                        auto idx)
  {
    if (faceGrid.size() != static_cast<size_t>(dimU) * dimV) return;
    if (lf == 0) // bottom (t = -1)
    {
      for (unsigned int j = 0; j < dimV && j < static_cast<unsigned int>(ny); ++j)
      {
        for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
        {
          const float3 target = faceGrid[j * dimU + i];
          const float3 base = nodal[idx(i, j, 0)];
          const float3 diff = make_float3(target.x - base.x, target.y - base.y, target.z - base.z);
          for (int k = 0; k < nz; ++k)
          {
            const float w = 0.5f * (1.0f - tNodes[k]);
            float3& p = nodal[idx(i, j, k)];
            p.x += diff.x * w;
            p.y += diff.y * w;
            p.z += diff.z * w;
          }
        }
      }
    }
    else if (lf == 5) // top (t = +1)
    {
      for (unsigned int j = 0; j < dimV && j < static_cast<unsigned int>(ny); ++j)
      {
        for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
        {
          const float3 target = faceGrid[j * dimU + i];
          const float3 base = nodal[idx(i, j, nz - 1)];
          const float3 diff = make_float3(target.x - base.x, target.y - base.y, target.z - base.z);
          for (int k = 0; k < nz; ++k)
          {
            const float w = 0.5f * (1.0f + tNodes[k]);
            float3& p = nodal[idx(i, j, k)];
            p.x += diff.x * w;
            p.y += diff.y * w;
            p.z += diff.z * w;
          }
        }
      }
    }
    else if (lf == 1) // front (s = -1)
    {
      for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
      {
        for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
        {
          const float3 target = faceGrid[k * dimU + i];
          const float3 base = nodal[idx(i, 0, k)];
          const float3 diff = make_float3(target.x - base.x, target.y - base.y, target.z - base.z);
          for (int j = 0; j < ny; ++j)
          {
            const float w = 0.5f * (1.0f - sNodes[j]);
            float3& p = nodal[idx(i, j, k)];
            p.x += diff.x * w;
            p.y += diff.y * w;
            p.z += diff.z * w;
          }
        }
      }
    }
    else if (lf == 3) // back (s = +1)
    {
      for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
      {
        for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
        {
          const float3 target = faceGrid[k * dimU + i];
          const float3 base = nodal[idx(i, ny - 1, k)];
          const float3 diff = make_float3(target.x - base.x, target.y - base.y, target.z - base.z);
          for (int j = 0; j < ny; ++j)
          {
            const float w = 0.5f * (1.0f + sNodes[j]);
            float3& p = nodal[idx(i, j, k)];
            p.x += diff.x * w;
            p.y += diff.y * w;
            p.z += diff.z * w;
          }
        }
      }
    }
    else if (lf == 2) // r = +1
    {
      for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
      {
        for (unsigned int j = 0; j < dimU && j < static_cast<unsigned int>(ny); ++j)
        {
          const float3 target = faceGrid[k * dimU + j];
          const float3 base = nodal[idx(nx - 1, j, k)];
          const float3 diff = make_float3(target.x - base.x, target.y - base.y, target.z - base.z);
          for (int i = 0; i < nx; ++i)
          {
            const float w = 0.5f * (1.0f + rNodes[i]);
            float3& p = nodal[idx(i, j, k)];
            p.x += diff.x * w;
            p.y += diff.y * w;
            p.z += diff.z * w;
          }
        }
      }
    }
    else if (lf == 4) // r = -1
    {
      for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
      {
        for (unsigned int j = 0; j < dimU && j < static_cast<unsigned int>(ny); ++j)
        {
          const float3 target = faceGrid[k * dimU + j];
          const float3 base = nodal[idx(0, j, k)];
          const float3 diff = make_float3(target.x - base.x, target.y - base.y, target.z - base.z);
          for (int i = 0; i < nx; ++i)
          {
            const float w = 0.5f * (1.0f - rNodes[i]);
            float3& p = nodal[idx(i, j, k)];
            p.x += diff.x * w;
            p.y += diff.y * w;
            p.z += diff.z * w;
          }
        }
      }
    }
  };

  unsigned int maxOrder = std::max({baseModes.x, baseModes.y, baseModes.z});
  for (int e : elem.edges)
  {
    auto it = curvedEdges.find(e);
    if (it != curvedEdges.end()) maxOrder = std::max(maxOrder, it->second.numPts);
  }
  for (int f : elem.faces)
  {
    auto it = curvedFaces.find(f);
    if (it != curvedFaces.end() && it->second.numPts > 0)
    {
      const unsigned int n = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(it->second.numPts))));
      maxOrder = std::max(maxOrder, n);
    }
  }
  geomModes = make_uint3(std::max(baseModes.x, maxOrder), std::max(baseModes.y, maxOrder), std::max(baseModes.z, maxOrder));
  if (geomModes.x == 0 || geomModes.y == 0 || geomModes.z == 0) return false;

  const int nx = static_cast<int>(geomModes.x);
  const int ny = static_cast<int>(geomModes.y);
  const int nz = static_cast<int>(geomModes.z);
  const std::vector<float> rNodes = GLLNodes(geomModes.x);
  const std::vector<float> sNodes = GLLNodes(geomModes.y);
  const std::vector<float> tNodes = GLLNodes(geomModes.z);
  auto idx = [&](int i, int j, int k) { return (static_cast<size_t>(k) * ny + j) * nx + i; };

  // Debug: surface mapping and whether each face is curved.
  std::printf("[Debug] Hex %d face mapping:\n", elem.id);
  for (int lf = 0; lf < 6; ++lf)
  {
    const int globalF = elem.faces[lf];
    const bool hasCurved = (curvedFaces.find(globalF) != curvedFaces.end());
    std::printf("  lf %d -> FACEID %d%s\n", lf, globalF, hasCurved ? " (CURVED)" : "");
  }

  // Map ShapeWeight(Z-order) -> XML/VTK vertex order so that the trilinear
  // corner weights apply to the correct physical vertices.
  static const int MATH_TO_XML_MAP[8] = {0, 1, 3, 2, 4, 5, 7, 6};
  float3 corners[8];
  for (int i = 0; i < 8; ++i)
  {
    const int xmlIdx = MATH_TO_XML_MAP[i];
    corners[i] = FetchVertex(vertices, elem.verts[xmlIdx]);
  }
  auto shapeWeight = [](int idx, float r, float s, float t) {
    const float rSign = (idx & 1) ? 1.0f : -1.0f;
    const float sSign = (idx & 2) ? 1.0f : -1.0f;
    const float tSign = (idx & 4) ? 1.0f : -1.0f;
    return 0.125f * (1.0f + rSign * r) * (1.0f + sSign * s) * (1.0f + tSign * t);
  };

  std::vector<float3> nodal(static_cast<size_t>(nx) * ny * nz, make_float3(0, 0, 0));
  for (int k = 0; k < nz; ++k)
  {
    for (int j = 0; j < ny; ++j)
    {
      for (int i = 0; i < nx; ++i)
      {
        const float r = rNodes[i];
        const float s = sNodes[j];
        const float t = tNodes[k];
        float3 p = make_float3(0, 0, 0);
        for (int c = 0; c < 8; ++c)
        {
          const float w = shapeWeight(c, r, s, t);
          p.x += w * corners[c].x;
          p.y += w * corners[c].y;
          p.z += w * corners[c].z;
        }
        nodal[idx(i, j, k)] = p;
      }
    }
  }

  const bool reversePlacement[12] = {false, false, true, true, false, false, false, false, false, false, true, true};
  for (int e = 0; e < 12; ++e)
  {
    const int globalE = elem.edges[e];
    if (globalE < 0) continue;
    auto ce = curvedEdges.find(globalE);
    if (ce == curvedEdges.end() || ce->second.pts.empty()) continue;
    auto eInfo = edges.find(globalE);
    bool reverse = reversePlacement[e];
    if (eInfo != edges.end())
    {
      const int localStart = elem.verts[hexEdgeVerts[e][0]];
      const int localEnd = elem.verts[hexEdgeVerts[e][1]];
      if (eInfo->second.v0 == localEnd && eInfo->second.v1 == localStart) reverse = !reverse;
    }
    unsigned int targetCount = (e == 1 || e == 3 || e == 9 || e == 11) ? geomModes.y
                            : (e == 4 || e == 5 || e == 6 || e == 7) ? geomModes.z
                            : geomModes.x;
    const std::vector<float3> edgePts = ResampleEdgePoints(ce->second.pts,
                                                           targetCount,
                                                           reverse,
                                                           ce->second.nodeType,
                                                           NodeDistribution::GLL);
    if (edgePts.size() != targetCount) continue;
    switch (e)
    {
      case 0: for (int i = 0; i < nx; ++i) nodal[idx(i, 0, 0)] = edgePts[i]; break;
      case 1: for (int j = 0; j < ny; ++j) nodal[idx(nx - 1, j, 0)] = edgePts[j]; break;
      case 2: for (int i = 0; i < nx; ++i) nodal[idx(i, ny - 1, 0)] = edgePts[i]; break;
      case 3: for (int j = 0; j < ny; ++j) nodal[idx(0, j, 0)] = edgePts[j]; break;
      case 4: for (int k = 0; k < nz; ++k) nodal[idx(0, 0, k)] = edgePts[k]; break;
      case 5: for (int k = 0; k < nz; ++k) nodal[idx(nx - 1, 0, k)] = edgePts[k]; break;
      case 6: for (int k = 0; k < nz; ++k) nodal[idx(nx - 1, ny - 1, k)] = edgePts[k]; break;
      case 7: for (int k = 0; k < nz; ++k) nodal[idx(0, ny - 1, k)] = edgePts[k]; break;
      case 8: for (int i = 0; i < nx; ++i) nodal[idx(i, 0, nz - 1)] = edgePts[i]; break;
      case 9: for (int j = 0; j < ny; ++j) nodal[idx(nx - 1, j, nz - 1)] = edgePts[j]; break;
      case 10: for (int i = 0; i < nx; ++i) nodal[idx(i, ny - 1, nz - 1)] = edgePts[i]; break;
      case 11: for (int j = 0; j < ny; ++j) nodal[idx(0, j, nz - 1)] = edgePts[j]; break;
    }
  }

  for (int lf = 0; lf < 6; ++lf)
  {
    const int globalF = elem.faces[lf];
    auto cf = curvedFaces.find(globalF);
    if (cf == curvedFaces.end() || cf->second.pts.empty()) continue;
    unsigned int dimU = 0, dimV = 0;
    if (lf == 0 || lf == 5) { dimU = geomModes.x; dimV = geomModes.y; }      // z faces
    else if (lf == 1 || lf == 3) { dimU = geomModes.x; dimV = geomModes.z; } // s faces
    else if (lf == 2 || lf == 4) { dimU = geomModes.y; dimV = geomModes.z; } // r faces
    else { dimU = geomModes.x; dimV = geomModes.y; }
    const std::array<float3, 4> cornersArr = face_corners(lf);
    std::vector<float3> faceGrid = ResampleQuadFaceLegendre(cf->second.pts,
                                                            dimU,
                                                            dimV,
                                                            cornersArr,
                                                            cf->second.nodeType,
                                                            NodeDistribution::GLL);
    if (faceGrid.empty())
    {
      faceGrid = ResampleQuadFace(cf->second.pts,
                                  dimU,
                                  dimV,
                                  cornersArr,
                                  cf->second.nodeType,
                                  NodeDistribution::GLL);
    }
    blend_face(lf, faceGrid, dimU, dimV, nodal, rNodes, sNodes, tNodes, nx, ny, nz, idx);
  }

  if (!TensorModalFit3D(nodal, rNodes, sNodes, tNodes, gx, gy, gz)) return false;
  return true;
}

bool BuildQuadGeometry(const ElemQuad& elem,
                       const std::vector<float3>& vertices,
                       const std::unordered_map<long long, int>& edgeLookup,
                       const std::unordered_map<int, CurvedEdge>& curvedEdges,
                       const std::unordered_map<int, CurvedFace>& curvedFaces,
                       const uint2& baseModes,
                       std::vector<float>& gx,
                       std::vector<float>& gy,
                       std::vector<float>& gz,
                       uint2& geomModes)
{
  unsigned int maxOrder = std::max(baseModes.x, baseModes.y);
  for (int e = 0; e < 4; ++e)
  {
    int va = elem.verts[e];
    int vb = elem.verts[(e + 1) % 4];
    auto itEdge = edgeLookup.find(EdgeKey(va, vb));
    if (itEdge != edgeLookup.end())
    {
      auto ce = curvedEdges.find(itEdge->second);
      if (ce != curvedEdges.end()) maxOrder = std::max(maxOrder, ce->second.numPts);
    }
  }

  auto cfIt = curvedFaces.find(elem.id);
  if (cfIt != curvedFaces.end() && !cfIt->second.pts.empty())
  {
    const unsigned int n = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(cfIt->second.pts.size()))));
    if (n > 1) maxOrder = std::max(maxOrder, n);
  }
  geomModes = make_uint2(std::max(baseModes.x, maxOrder), std::max(baseModes.y, maxOrder));
  const int nx = static_cast<int>(geomModes.x);
  const int ny = static_cast<int>(geomModes.y);
  const std::vector<float> rNodes = GLLNodes(geomModes.x);
  const std::vector<float> sNodes = GLLNodes(geomModes.y);
  auto idx = [&](int i, int j) { return j * nx + i; };

  float3 c0 = FetchVertex(vertices, elem.verts[0]);
  float3 c1 = FetchVertex(vertices, elem.verts[1]);
  float3 c2 = FetchVertex(vertices, elem.verts[2]);
  float3 c3 = FetchVertex(vertices, elem.verts[3]);

  if (cfIt != curvedFaces.end() && !cfIt->second.pts.empty())
  {
    std::array<float3, 4> corners = {c0, c1, c2, c3};
    std::vector<float3> faceGrid = ResampleQuadFace(cfIt->second.pts,
                                                    static_cast<unsigned int>(nx),
                                                    static_cast<unsigned int>(ny),
                                                    corners,
                                                    cfIt->second.nodeType,
                                                    NodeDistribution::GLL);
    if (faceGrid.size() == static_cast<size_t>(nx * ny))
    {
      return TensorModalFit2D(faceGrid, rNodes, sNodes, gx, gy, gz);
    }
  }

  std::array<std::vector<float3>, 4> edgesInterp;
  edgesInterp[0].resize(nx);
  edgesInterp[2].resize(nx);
  edgesInterp[1].resize(ny);
  edgesInterp[3].resize(ny);
  for (int i = 0; i < nx; ++i)
  {
    const float u = (nx == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(nx - 1);
    edgesInterp[0][i] = make_float3(c0.x + u * (c1.x - c0.x),
                                    c0.y + u * (c1.y - c0.y),
                                    c0.z + u * (c1.z - c0.z));
    edgesInterp[2][i] = make_float3(c3.x + u * (c2.x - c3.x),
                                    c3.y + u * (c2.y - c3.y),
                                    c3.z + u * (c2.z - c3.z));
  }
  for (int j = 0; j < ny; ++j)
  {
    const float v = (ny == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(ny - 1);
    edgesInterp[3][j] = make_float3(c0.x + v * (c3.x - c0.x),
                                    c0.y + v * (c3.y - c0.y),
                                    c0.z + v * (c3.z - c0.z));
    edgesInterp[1][j] = make_float3(c1.x + v * (c2.x - c1.x),
                                    c1.y + v * (c2.y - c1.y),
                                    c1.z + v * (c2.z - c1.z));
  }

  for (int e = 0; e < 4; ++e)
  {
    int va = elem.verts[e];
    int vb = elem.verts[(e + 1) % 4];
    auto itEdge = edgeLookup.find(EdgeKey(va, vb));
    if (itEdge == edgeLookup.end()) continue;
    auto ce = curvedEdges.find(itEdge->second);
    if (ce == curvedEdges.end() || ce->second.pts.empty()) continue;
    const bool reverse = (vb < va);
    std::vector<float3> pts = ResampleEdgePoints(ce->second.pts,
                                                 (e % 2 == 0) ? geomModes.x : geomModes.y,
                                                 reverse,
                                                 ce->second.nodeType,
                                                 NodeDistribution::GLL);
    if (pts.empty()) continue;
    if (e == 0) edgesInterp[0] = pts;
    else if (e == 2) edgesInterp[2] = pts;
    else if (e == 1) edgesInterp[1] = pts;
    else edgesInterp[3] = pts;
  }

  std::vector<float3> nodal(static_cast<size_t>(nx) * ny);
  for (int j = 0; j < ny; ++j)
  {
    const float v = (ny == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(ny - 1);
    for (int i = 0; i < nx; ++i)
    {
      const float u = (nx == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(nx - 1);
      const float3 Cb = edgesInterp[0][i];
      const float3 Ct = edgesInterp[2][i];
      const float3 Cl = edgesInterp[3][j];
      const float3 Cr = edgesInterp[1][j];
      float3 cornerBlend = make_float3(
        (1 - u) * (1 - v) * c0.x + u * (1 - v) * c1.x + u * v * c2.x + (1 - u) * v * c3.x,
        (1 - u) * (1 - v) * c0.y + u * (1 - v) * c1.y + u * v * c2.y + (1 - u) * v * c3.y,
        (1 - u) * (1 - v) * c0.z + u * (1 - v) * c1.z + u * v * c2.z + (1 - u) * v * c3.z);
      float3 p = make_float3(
        (1 - v) * Cb.x + v * Ct.x + (1 - u) * Cl.x + u * Cr.x - cornerBlend.x,
        (1 - v) * Cb.y + v * Ct.y + (1 - u) * Cl.y + u * Cr.y - cornerBlend.y,
        (1 - v) * Cb.z + v * Ct.z + (1 - u) * Cl.z + u * Cr.z - cornerBlend.z);
      nodal[idx(i, j)] = p;
    }
  }
  return TensorModalFit2D(nodal, rNodes, sNodes, gx, gy, gz);
}

bool BuildTriGeometry(const ElemTri& elem,
                      const std::vector<float3>& vertices,
                      const std::unordered_map<long long, int>& edgeLookup,
                      const std::unordered_map<int, EdgeInfo>& edges,
                      const std::unordered_map<int, CurvedEdge>& curvedEdges,
                      const uint2& baseModes,
                      std::vector<float>& gx,
                      std::vector<float>& gy,
                      std::vector<float>& gz,
                      uint2& geomModes)
{
  unsigned int maxOrder = std::max(baseModes.x, baseModes.y);
  const int triEdge[3][2] = {{0,1},{1,2},{2,0}};
  for (int e = 0; e < 3; ++e)
  {
    int a = elem.verts[triEdge[e][0]];
    int b = elem.verts[triEdge[e][1]];
    auto itE = edgeLookup.find(EdgeKey(a, b));
    if (itE != edgeLookup.end())
    {
      auto ce = curvedEdges.find(itE->second);
      if (ce != curvedEdges.end()) maxOrder = std::max(maxOrder, ce->second.numPts);
    }
  }
  const unsigned int order = std::max(2u, maxOrder);
  geomModes = make_uint2(order, order);

  const size_t nPts = static_cast<size_t>(order) * (order + 1) / 2;
  struct NodeRef { unsigned int i; unsigned int j; size_t idx; };
  std::vector<NodeRef> nodeRefs;
  nodeRefs.reserve(nPts);
  std::vector<float2> nodesRS;
  nodesRS.reserve(nPts);
  for (unsigned int j = 0; j < order; ++j)
  {
    for (unsigned int i = 0; i + j < order; ++i)
    {
      const float u = (order == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(order - 1);
      const float v = (order == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(order - 1);
      nodesRS.push_back(make_float2(2.0f * u - 1.0f, 2.0f * v - 1.0f));
      nodeRefs.push_back({i, j, nodeRefs.size()});
    }
  }

  const float3 v0 = FetchVertex(vertices, elem.verts[0]);
  const float3 v1 = FetchVertex(vertices, elem.verts[1]);
  const float3 v2 = FetchVertex(vertices, elem.verts[2]);
  std::vector<float3> nodal(nPts);
  for (size_t idx = 0; idx < nPts; ++idx)
  {
    const float r = nodesRS[idx].x;
    const float s = nodesRS[idx].y;
    const float u = 0.5f * (r + 1.0f);
    const float v = 0.5f * (s + 1.0f);
    const float w = std::max(0.0f, 1.0f - u - v);
    nodal[idx] = make_float3(
      w * v0.x + u * v1.x + v * v2.x,
      w * v0.y + u * v1.y + v * v2.y,
      w * v0.z + u * v1.z + v * v2.z);
  }

  auto set_edge = [&](int localEdge, const std::vector<float3>& samples)
  {
    if (samples.empty()) return;
    std::vector<size_t> idxList;
    if (localEdge == 0)
    {
      for (auto& nr : nodeRefs) if (nr.j == 0) idxList.push_back(nr.idx);
    }
    else if (localEdge == 1)
    {
      for (auto& nr : nodeRefs) if (nr.i + nr.j == order - 1) idxList.push_back(nr.idx);
    }
    else if (localEdge == 2)
    {
      for (auto& nr : nodeRefs) if (nr.i == 0) idxList.push_back(nr.idx);
    }
    if (idxList.empty()) return;
    std::sort(idxList.begin(), idxList.end());
    std::vector<float3> resampled = ResampleEdgePoints(samples, static_cast<unsigned int>(idxList.size()));
    if (resampled.size() == idxList.size())
    {
      for (size_t k = 0; k < idxList.size(); ++k) nodal[idxList[k]] = resampled[k];
    }
  };

  for (int e = 0; e < 3; ++e)
  {
    int a = elem.verts[triEdge[e][0]];
    int b = elem.verts[triEdge[e][1]];
    auto itE = edgeLookup.find(EdgeKey(a, b));
    if (itE == edgeLookup.end()) continue;
    auto ce = curvedEdges.find(itE->second);
    if (ce == curvedEdges.end()) continue;
    bool reverse = false;
    auto eInfo = edges.find(itE->second);
    if (eInfo != edges.end())
    {
      if (eInfo->second.v0 == b && eInfo->second.v1 == a) reverse = true;
    }
    std::vector<float3> samples = ResampleEdgePoints(ce->second.pts, order, reverse, ce->second.nodeType);
    set_edge(e, samples);
  }

  return SolveTriModalCoeffs(nodal, nodesRS, geomModes, gx, gy, gz);
}

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
                        uint3& geomModes)
{
  unsigned int maxOrder = std::max({baseModes.x, baseModes.y, baseModes.z});
  const int edgePairs[9][2] = {{0,1},{1,2},{2,0},{3,4},{4,5},{5,3},{0,3},{1,4},{2,5}};
  for (int e = 0; e < 9; ++e)
  {
    int globalE = elem.edges[e];
    if (globalE < 0)
    {
      int a = elem.verts[edgePairs[e][0]];
      int b = elem.verts[edgePairs[e][1]];
      auto itE = edgeLookup.find(EdgeKey(a, b));
      if (itE != edgeLookup.end()) globalE = itE->second;
    }
    auto itC = curvedEdges.find(globalE);
    if (itC != curvedEdges.end()) maxOrder = std::max(maxOrder, itC->second.numPts);
  }
  auto tri_order_from_pts = [](unsigned int n)->unsigned int {
    if (n == 0) return 0;
    double o = (std::sqrt(8.0 * static_cast<double>(n) + 1.0) - 1.0) * 0.5;
    return static_cast<unsigned int>(std::round(o));
  };
  auto quad_order_from_pts = [](unsigned int n)->unsigned int {
    if (n == 0) return 0;
    double o = std::sqrt(static_cast<double>(n));
    return static_cast<unsigned int>(std::round(o));
  };
  for (int f = 0; f < 5; ++f)
  {
    int fid = elem.faces[f];
    if (fid < 0) continue;
    auto cf = curvedFaces.find(fid);
    if (cf == curvedFaces.end() || cf->second.numPts == 0) continue;
    unsigned int o = (f < 2) ? tri_order_from_pts(cf->second.numPts) : quad_order_from_pts(cf->second.numPts);
    if (o > 1) maxOrder = std::max(maxOrder, o);
  }
  geomModes = make_uint3(std::max(1u, maxOrder),
                         std::max(1u, maxOrder),
                         std::max(maxOrder, std::max(1u, baseModes.x)));

  const unsigned int nx = geomModes.x;
  const unsigned int ny = geomModes.y;
  const unsigned int nz = geomModes.z;
  const size_t nCoeffs = PrismCoeffCount(geomModes);
  if (nCoeffs == 0) return false;

  const std::vector<float> rNodes = GLLNodes(nx);
  const std::vector<float> sNodes = GLLNodes(ny);
  const std::vector<float> tNodes = GLLNodes(nz);

  std::vector<float3> nodal;
  nodal.reserve(nCoeffs);
  std::vector<uint64_t> nodeKeys;
  nodeKeys.reserve(nCoeffs);
  std::unordered_map<uint64_t, size_t, PrismNodeKeyHasher> keyToIdx;
  std::array<float3,6> v{};
  for (int i = 0; i < 6; ++i) v[i] = FetchVertex(vertices, elem.verts[i]);

  for (unsigned int ir = 0; ir < nx; ++ir)
  {
    for (unsigned int js = 0; js < ny; ++js)
    {
      unsigned int maxK = (nz > ir) ? (nz - ir) : 0;
      for (unsigned int kt = 0; kt < maxK; ++kt)
      {
        const float r = rNodes[ir];
        const float s = sNodes[js];
        const float t = tNodes[kt];
        const uint64_t key = PrismNodeKey(ir, js, kt);
        nodeKeys.push_back(key);
        keyToIdx[key] = nodal.size();
        nodal.push_back(ReferenceToWorldPrismLinear(v, make_float3(r, s, t)));
      }
    }
  }

  auto apply_edge = [&](int localEdge, const std::vector<float3>& samples, NodeDistribution srcType)
  {
    if (samples.empty()) return;
    std::vector<size_t> idxList;
    switch (localEdge)
    {
      case 0:
        for (unsigned int ir = 0; ir < nx; ++ir)
          if (keyToIdx.count(PrismNodeKey(ir, 0, 0)))
            idxList.push_back(keyToIdx[PrismNodeKey(ir, 0, 0)]);
        break;
      case 1:
        for (unsigned int js = 0; js < ny; ++js)
          if (keyToIdx.count(PrismNodeKey(nx - 1, js, 0)))
            idxList.push_back(keyToIdx[PrismNodeKey(nx - 1, js, 0)]);
        break;
      case 6:
        for (unsigned int kt = 0; kt < nz; ++kt)
          if (keyToIdx.count(PrismNodeKey(0, 0, kt)))
            idxList.push_back(keyToIdx[PrismNodeKey(0, 0, kt)]);
        break;
      case 7:
        for (unsigned int kt = 0; kt < nz - (nx - 1); ++kt)
          if (keyToIdx.count(PrismNodeKey(nx - 1, 0, kt)))
            idxList.push_back(keyToIdx[PrismNodeKey(nx - 1, 0, kt)]);
        break;
      case 8:
        for (unsigned int kt = 0; kt < nz - (nx - 1); ++kt)
          if (keyToIdx.count(PrismNodeKey(nx - 1, ny - 1, kt)))
            idxList.push_back(keyToIdx[PrismNodeKey(nx - 1, ny - 1, kt)]);
        break;
      default:
        return;
    }
    if (idxList.empty()) return;
    std::vector<float3> resampled = ResampleEdgePoints(samples,
                                                       static_cast<unsigned int>(idxList.size()),
                                                       false,
                                                       srcType,
                                                       NodeDistribution::GLL);
    if (resampled.size() == idxList.size())
    {
      for (size_t i = 0; i < idxList.size(); ++i) nodal[idxList[i]] = resampled[i];
    }
  };

  for (int e = 0; e < 9; ++e)
  {
    int globalE = elem.edges[e];
    if (globalE < 0)
    {
      int a = elem.verts[edgePairs[e][0]];
      int b = elem.verts[edgePairs[e][1]];
      auto itE = edgeLookup.find(EdgeKey(a, b));
      if (itE != edgeLookup.end()) globalE = itE->second;
    }
    if (globalE < 0) continue;
    auto itC = curvedEdges.find(globalE);
    if (itC == curvedEdges.end()) continue;
    apply_edge(e, itC->second.pts, itC->second.nodeType);
  }

  auto apply_tri_face = [&](int faceIdx, bool top)
  {
    int fid = elem.faces[faceIdx];
    if (fid < 0) return;
    auto cf = curvedFaces.find(fid);
    if (cf == curvedFaces.end() || cf->second.pts.empty()) return;
    unsigned int order = std::min({geomModes.x, geomModes.y});
    if (order < 2) return;
    std::array<float3,3> corners = top ?
      std::array<float3,3>{FetchVertex(vertices, elem.verts[3]), FetchVertex(vertices, elem.verts[4]), FetchVertex(vertices, elem.verts[5])} :
      std::array<float3,3>{FetchVertex(vertices, elem.verts[0]), FetchVertex(vertices, elem.verts[1]), FetchVertex(vertices, elem.verts[2])};
    std::vector<float3> triGrid = ResampleTriFace(cf->second.pts, order, corners, cf->second.nodeType, NodeDistribution::GLL);
    if (triGrid.size() < static_cast<size_t>(order) * (order + 1) / 2) return;
    size_t idxTri = 0;
    for (unsigned int j = 0; j < order; ++j)
    {
      for (unsigned int i = 0; i + j < order; ++i, ++idxTri)
      {
        unsigned int ir = i;
        unsigned int js = j;
        unsigned int kt = top ? ((nz > ir) ? nz - ir - 1 : 0) : 0;
        uint64_t key = PrismNodeKey(ir, js, kt);
        auto itKey = keyToIdx.find(key);
        if (itKey != keyToIdx.end()) nodal[itKey->second] = triGrid[idxTri];
      }
    }
  };
  apply_tri_face(0, false);
  apply_tri_face(1, true);

  auto apply_quad_face = [&](int faceIdx,
                             unsigned int dimU,
                             unsigned int dimV,
                             int fixedI,
                             int fixedJ,
                             const std::array<float3,4>& corners)
  {
    int fid = elem.faces[faceIdx];
    if (fid < 0 || dimU == 0 || dimV == 0) return;
    auto cf = curvedFaces.find(fid);
    if (cf == curvedFaces.end() || cf->second.pts.empty()) return;
    std::vector<float3> faceGrid = ResampleQuadFace(cf->second.pts, dimU, dimV, corners, cf->second.nodeType);
    if (faceGrid.size() != static_cast<size_t>(dimU) * dimV) return;
    for (unsigned int v = 0; v < dimV; ++v)
    {
      for (unsigned int u = 0; u < dimU; ++u)
      {
        size_t idx = static_cast<size_t>(v) * dimU + u;
        int i = fixedI >= 0 ? fixedI : static_cast<int>(u);
        int j = fixedJ >= 0 ? fixedJ : static_cast<int>(u);
        if (fixedI >= 0 && fixedJ < 0) j = static_cast<int>(u);
        if (fixedI < 0 && fixedJ >= 0) i = static_cast<int>(u);
        int k = static_cast<int>(v);
        if (i < 0 || j < 0 || k < 0 || i >= (int)nx || j >= (int)ny) continue;
        unsigned int maxK = (nz > static_cast<unsigned int>(i)) ? (nz - static_cast<unsigned int>(i)) : 0;
        if (static_cast<unsigned int>(k) >= maxK) continue;
        auto itKey = keyToIdx.find(PrismNodeKey(static_cast<unsigned int>(i),
                                                static_cast<unsigned int>(j),
                                                static_cast<unsigned int>(k)));
        if (itKey != keyToIdx.end()) nodal[itKey->second] = faceGrid[idx];
      }
    }
  };
  apply_quad_face(2, nx, nz, -1, 0,
                  {FetchVertex(vertices, elem.verts[0]), FetchVertex(vertices, elem.verts[1]),
                   FetchVertex(vertices, elem.verts[4]), FetchVertex(vertices, elem.verts[3])});
  apply_quad_face(3, ny, (nz > 0) ? nz - 1 : nz, (nx > 0) ? nx - 1 : 0, -1,
                  {FetchVertex(vertices, elem.verts[1]), FetchVertex(vertices, elem.verts[2]),
                   FetchVertex(vertices, elem.verts[5]), FetchVertex(vertices, elem.verts[4])});
  apply_quad_face(4, ny, nz, 0, -1,
                  {FetchVertex(vertices, elem.verts[2]), FetchVertex(vertices, elem.verts[0]),
                   FetchVertex(vertices, elem.verts[3]), FetchVertex(vertices, elem.verts[5])});

  std::vector<float> V(nCoeffs * nCoeffs);
  for (size_t row = 0; row < nCoeffs; ++row)
  {
    const uint64_t key = nodeKeys[row];
    const unsigned int ir = static_cast<unsigned int>(key >> 42);
    const unsigned int js = static_cast<unsigned int>((key >> 21) & 0x1FFFFF);
    const unsigned int kt = static_cast<unsigned int>(key & 0x1FFFFF);
    const float3 rst = make_float3(rNodes[ir], sNodes[js], tNodes[kt]);
    size_t col = 0;
    for (unsigned int i = 0; i < geomModes.x; ++i)
    {
      const unsigned int maxK = (geomModes.z > i) ? (geomModes.z - i) : 0;
      for (unsigned int j = 0; j < geomModes.y; ++j)
      {
        for (unsigned int k = 0; k < maxK; ++k)
        {
          V[row * nCoeffs + col] = EvaluatePrismBasis(i, j, k, rst);
          ++col;
        }
      }
    }
  }

  std::vector<float> V_inv;
  if (!InvertSquareMatrix(V, static_cast<int>(nCoeffs), V_inv)) return false;
  gx.assign(nCoeffs, 0.0f);
  gy.assign(nCoeffs, 0.0f);
  gz.assign(nCoeffs, 0.0f);
  for (size_t m = 0; m < nCoeffs; ++m)
  {
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (size_t n = 0; n < nCoeffs; ++n)
    {
      const float w = V_inv[m * nCoeffs + n];
      const float3 vpt = nodal[n];
      sx += w * vpt.x;
      sy += w * vpt.y;
      sz += w * vpt.z;
    }
    gx[m] = sx;
    gy[m] = sy;
    gz[m] = sz;
  }
  return true;
}
