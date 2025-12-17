#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "xml_fld_common.h"

struct EdgeInfo
{
  int v0{-1};
  int v1{-1};
};

struct FaceInfo
{
  int id{-1};
  char type{'Q'};
  std::vector<int> edges;
  std::vector<int> verts;
};

struct ElemHex
{
  int id{0};
  std::array<int, 8> verts{};
  std::array<int, 12> edges{};
  std::array<int, 6> faces{};
  std::array<bool, 12> edgeOrient{}; // true: v0->v1 与局部一致
  std::array<int, 6> faceOrient{};   // 0: 正向, 1: 反向, -1: 未知
  bool valid{false};
};

struct ElemPrism
{
  int id{0};
  std::array<int, 6> verts{};
  std::array<int, 9> edges{};
  std::array<int, 5> faces{};
  bool valid{false};
};

struct ElemQuad
{
  int id{0};
  std::array<int, 4> verts{};
};

struct ElemTri
{
  int id{0};
  std::array<int, 3> verts{};
};

struct CurvedEdge
{
  unsigned int numPts{0};
  std::vector<float3> pts;
  NodeDistribution nodeType{NodeDistribution::GLL};
};

struct CurvedFace
{
  unsigned int numPts{0};
  char type{'Q'};
  std::vector<float3> pts;
  NodeDistribution nodeType{NodeDistribution::GLL};
};

struct GeometryData
{
  std::vector<float3> vertices;
  std::unordered_map<int, EdgeInfo> edges;
  std::unordered_map<long long, int> edgeLookup;
  std::unordered_map<int, FaceInfo> faces;
  std::vector<ElemHex> hexes;
  std::vector<ElemPrism> prisms;
  std::vector<ElemQuad> quads;
  std::vector<ElemTri> tris;
  std::unordered_map<int, CurvedEdge> curvedEdges;
  std::unordered_map<int, CurvedFace> curvedFaces;
  uint3 defaultModes{make_uint3(1, 1, 1)};
};

long long EdgeKey(int a, int b);

bool ParseGeometry(const std::string& xmlText, GeometryData& out);
