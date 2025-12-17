#include "xml_fld_geometry_parser.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <sstream>
#include <unordered_set>
#include <type_traits>
#include <cctype>

#include "xml_fld_common.h"

namespace
{
// Remove XML comment blocks <!-- ... --> to avoid parsing commented-out tags.
std::string StripXmlComments(const std::string& text)
{
  std::string out;
  out.reserve(text.size());
  size_t pos = 0;
  while (pos < text.size())
  {
    size_t start = text.find("<!--", pos);
    if (start == std::string::npos)
    {
      out.append(text.substr(pos));
      break;
    }
    out.append(text.substr(pos, start - pos));
    size_t end = text.find("-->", start + 4);
    if (end == std::string::npos)
    {
      // Unclosed comment; drop the rest.
      break;
    }
    pos = end + 3;
  }
  return out;
}
} // namespace

long long EdgeKey(int a, int b)
{
  if (a > b) std::swap(a, b);
  return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
}

bool ParseGeometry(const std::string& xmlText, GeometryData& out)
{
  out = GeometryData{};
  const std::string xml = StripXmlComments(xmlText);

  // vertices
  {
    size_t pos = 0;
    while (true)
    {
      size_t open = xml.find("<V", pos);
      if (open == std::string::npos) break;
      size_t gt = xml.find('>', open);
      if (gt == std::string::npos) break;
      size_t close = xml.find("</V>", gt);
      if (close == std::string::npos) break;

      const std::string tagHeader = xml.substr(open, gt - open + 1);
      const std::string body = xml.substr(gt + 1, close - gt - 1);
      int id = static_cast<int>(out.vertices.size());
      const std::string idStr = GetAttribute(tagHeader, "ID");
      if (!idStr.empty()) id = std::atoi(idStr.c_str());
      float3 v = make_float3(0, 0, 0);
      std::istringstream iss(body);
      iss >> v.x >> v.y >> v.z;
      if (id >= static_cast<int>(out.vertices.size())) out.vertices.resize(id + 1);
      out.vertices[id] = v;
      pos = close + 4;
    }
  }

  if (out.vertices.empty()) return false;

  const size_t elemBlockStart = xml.find("<ELEMENT");
  const size_t elemBlockEnd = xml.find("</ELEMENT>");
  const std::string elementSection = (elemBlockStart != std::string::npos &&
                                      elemBlockEnd != std::string::npos &&
                                      elemBlockEnd > elemBlockStart)
                                       ? xml.substr(elemBlockStart, elemBlockEnd - elemBlockStart)
                                       : xml;

  // default modes
  {
    const size_t expPos = xml.find("NUMMODES");
    if (expPos != std::string::npos)
    {
      size_t tagStart = xml.rfind('<', expPos);
      size_t tagEnd = xml.find('>', expPos);
      if (tagStart != std::string::npos && tagEnd != std::string::npos && tagEnd > tagStart)
      {
        const std::string tag = xml.substr(tagStart, tagEnd - tagStart + 1);
        const std::string numModesStr = GetAttribute(tag, "NUMMODES");
        if (!numModesStr.empty()) out.defaultModes = ParseModes(numModesStr);
      }
    }
    if (out.defaultModes.x == 0) out.defaultModes = make_uint3(1, 1, 1);
  }

  // edges
  {
    size_t blockStart = xml.find("<EDGE");
    size_t blockEnd = xml.find("</EDGE>");
    if (blockStart != std::string::npos && blockEnd != std::string::npos && blockEnd > blockStart)
    {
      const std::string block = xml.substr(blockStart, blockEnd - blockStart);
      std::regex edge_re(R"(<E\s+ID=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</E>)", std::regex::icase);
      auto begin = std::sregex_iterator(block.begin(), block.end(), edge_re);
      auto end = std::sregex_iterator();
      for (auto it = begin; it != end; ++it)
      {
        const int id = std::atoi((*it)[1].str().c_str());
        std::istringstream iss((*it)[2].str());
        EdgeInfo e;
        if (iss >> e.v0 >> e.v1) out.edges[id] = e;
      }
    }
    for (const auto& kv : out.edges)
    {
      out.edgeLookup[EdgeKey(kv.second.v0, kv.second.v1)] = kv.first;
    }
  }

  // faces
  {
    size_t faceStart = xml.find("<FACE");
    size_t faceEnd = xml.find("</FACE>");
    if (faceStart != std::string::npos && faceEnd != std::string::npos && faceEnd > faceStart)
    {
      const std::string block = xml.substr(faceStart, faceEnd - faceStart);
      std::regex face_re(R"(<(Q|T)\s+ID=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</\1>)", std::regex::icase);
      auto begin = std::sregex_iterator(block.begin(), block.end(), face_re);
      auto end = std::sregex_iterator();
      for (auto it = begin; it != end; ++it)
      {
        FaceInfo f;
        f.type = static_cast<char>(std::toupper((*it)[1].str()[0]));
        f.id = std::atoi((*it)[2].str().c_str());
        std::istringstream iss((*it)[3].str());
        int eid;
        while (iss >> eid) f.edges.push_back(eid);
        if (!f.edges.empty())
        {
          auto eIt = out.edges.find(f.edges.front());
          if (eIt != out.edges.end())
          {
            f.verts.push_back(eIt->second.v0);
            f.verts.push_back(eIt->second.v1);
            for (size_t idx = 1; idx < f.edges.size(); ++idx)
            {
              auto e2 = out.edges.find(f.edges[idx]);
              if (e2 == out.edges.end()) continue;
              int a = e2->second.v0;
              int b = e2->second.v1;
              const int last = f.verts.back();
              if (a == last)
                f.verts.push_back(b);
              else if (b == last)
                f.verts.push_back(a);
              else if (a == f.verts.front())
                f.verts.insert(f.verts.begin(), b);
              else if (b == f.verts.front())
                f.verts.insert(f.verts.begin(), a);
            }
          }
          out.faces[f.id] = f;
        }
      }
    }
  }

  auto init_array = [](auto& arr) { for (auto& v : arr) v = -1; };

  const int hexEdgeVerts[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {0, 4}, {1, 5}, {2, 6}, {3, 7},
      {4, 5}, {5, 6}, {6, 7}, {7, 4}};

  auto build_hex_from_faces = [&](const std::array<int, 6>& faceIds, ElemHex& outHex) -> bool
  {
    outHex.faces = faceIds;
    auto face_vertices = [&](int fid) -> std::vector<int>
    {
      std::vector<int> result;
      auto it = out.faces.find(fid);
      if (it == out.faces.end() || it->second.verts.empty()) return result;
      for (int v : it->second.verts)
      {
        if (std::find(result.begin(), result.end(), v) == result.end())
        {
          result.push_back(v);
          if (result.size() == 4) break;
        }
      }
      return result;
    };

    std::array<std::vector<int>, 6> fVerts;
    bool haveAllFaces = true;
    for (int i = 0; i < 6; ++i)
    {
      fVerts[i] = face_vertices(faceIds[i]);
      if (fVerts[i].size() < 3) haveAllFaces = false;
    }
    auto find_common = [&](int a, int b, int c) -> int
    {
      for (int va : fVerts[a])
      {
        if (std::find(fVerts[b].begin(), fVerts[b].end(), va) != fVerts[b].end() &&
            std::find(fVerts[c].begin(), fVerts[c].end(), va) != fVerts[c].end())
        {
          return va;
        }
      }
      return -1;
    };
    if (haveAllFaces)
    {
      std::array<int, 8> canon{};
      canon[0] = find_common(0, 1, 4);
      canon[1] = find_common(0, 1, 2);
      canon[2] = find_common(0, 2, 3);
      canon[3] = find_common(0, 3, 4);
      canon[4] = find_common(1, 4, 5);
      canon[5] = find_common(1, 2, 5);
      canon[6] = find_common(2, 3, 5);
      canon[7] = find_common(3, 4, 5);
      if (std::all_of(canon.begin(), canon.end(), [](int v) { return v >= 0; }))
      {
        for (int i = 0; i < 8; ++i) outHex.verts[i] = canon[i];
      }
    }

    if (std::any_of(outHex.verts.begin(), outHex.verts.end(), [](int v) { return v < 0; }))
      return false;

    // 重新建立边（基于当前 verts），同时记录方向
    for (int e = 0; e < 12; ++e)
    {
      const int a = outHex.verts[hexEdgeVerts[e][0]];
      const int b = outHex.verts[hexEdgeVerts[e][1]];
      outHex.edges[e] = -1;
      outHex.edgeOrient[e] = true;
      auto it = out.edgeLookup.find(EdgeKey(a, b));
      if (it != out.edgeLookup.end())
      {
        outHex.edges[e] = it->second;
        auto edgeIt = out.edges.find(it->second);
        if (edgeIt != out.edges.end())
        {
          const EdgeInfo& phys = edgeIt->second;
          outHex.edgeOrient[e] = (phys.v0 == a && phys.v1 == b);
        }
      }
    }

    // 计算面朝向（Nektar Orientation：0..7，含转置），默认 -1
    outHex.faceOrient.fill(-1);
    const int stdHexFaceVerts[6][4] = {
        {0, 1, 2, 3}, {0, 1, 5, 4}, {1, 2, 6, 5},
        {3, 2, 6, 7}, {0, 3, 7, 4}, {4, 5, 6, 7}};

    for (int f = 0; f < 6; ++f)
    {
      int physFid = outHex.faces[f];
      auto itF = out.faces.find(physFid);
      if (itF == out.faces.end() || itF->second.verts.size() < 4) continue;

      const auto& physVerts = itF->second.verts;
      const int N = 4; // Quad faces

      int h0 = outHex.verts[stdHexFaceVerts[f][0]];
      int h1 = outHex.verts[stdHexFaceVerts[f][1]]; // dir1
      int h3 = outHex.verts[stdHexFaceVerts[f][3]]; // dir2

      auto it0 = std::find(physVerts.begin(), physVerts.end(), h0);
      if (it0 == physVerts.end()) continue;
      int idx0 = static_cast<int>(std::distance(physVerts.begin(), it0));

      auto find_step = [&](int v) -> int {
        auto it = std::find(physVerts.begin(), physVerts.end(), v);
        if (it == physVerts.end()) return -1;
        int idx = static_cast<int>(std::distance(physVerts.begin(), it));
        return (idx - idx0 + N) % N;
      };

      int step1 = find_step(h1); // h0 -> h1
      int step3 = find_step(h3); // h0 -> h3
      if (step1 < 0 || step3 < 0) continue;

      // 合法的四边形映射步长只能是 1 或 3（相邻）
      if (step1 == 1 && step3 == 3)
      {
        // 逆时针一致
        switch (idx0)
        {
          case 0: outHex.faceOrient[f] = 0; break; // Standard
          case 1: outHex.faceOrient[f] = 5; break; // Transposed (Rot +90)
          case 2: outHex.faceOrient[f] = 3; break; // 旋转 180
          case 3: outHex.faceOrient[f] = 6; break; // Transposed (Rot -90)
        }
      }
      else if (step1 == 3 && step3 == 1)
      {
        // 顺时针
        switch (idx0)
        {
          case 0: outHex.faceOrient[f] = 4; break; // Transposed
          case 1: outHex.faceOrient[f] = 1; break; // Dir1 Fwd, Dir2 Bwd
          case 2: outHex.faceOrient[f] = 7; break; // Transposed 180
          case 3: outHex.faceOrient[f] = 2; break; // Dir1 Bwd, Dir2 Fwd
        }
      }
      // (1,1) 或 (3,3) 不应出现
    }

    outHex.valid = true;
    return true;
  };

  auto parse_hex_tags = [&](const std::string& tag)
  {
    const std::string openTag = "<" + tag;
    const std::string closeTag = "</" + tag + ">";
    size_t pos = 0;
    while (true)
    {
      size_t open = elementSection.find(openTag, pos);
      if (open == std::string::npos) break;
      size_t gt = elementSection.find('>', open);
      if (gt == std::string::npos) break;
      size_t close = elementSection.find(closeTag, gt);
      if (close == std::string::npos) break;

      const std::string header = elementSection.substr(open, gt - open + 1);
      const std::string body = elementSection.substr(gt + 1, close - gt - 1);
      ElemHex hex{};
      init_array(hex.verts);
      init_array(hex.edges);
      hex.faces.fill(-1);
      hex.id = static_cast<int>(out.hexes.size());
      const std::string idStr = GetAttribute(header, "ID");
      if (!idStr.empty()) hex.id = std::atoi(idStr.c_str());

      std::istringstream iss(body);
      std::vector<int> vals;
      int v;
      while (iss >> v) vals.push_back(v);
      if (vals.size() == 6)
      {
        std::array<int, 6> faceIds{};
        for (int i = 0; i < 6; ++i) faceIds[i] = vals[i];
        build_hex_from_faces(faceIds, hex);
      }
      else if (vals.size() == 8)
      {
        for (int i = 0; i < 8; ++i) hex.verts[i] = vals[i];
        for (int e = 0; e < 12; ++e)
        {
          auto it = out.edgeLookup.find(EdgeKey(hex.verts[hexEdgeVerts[e][0]], hex.verts[hexEdgeVerts[e][1]]));
          hex.edges[e] = (it != out.edgeLookup.end()) ? it->second : -1;
        }
        hex.valid = true;
      }
      if (hex.valid) out.hexes.push_back(hex);
      pos = close + closeTag.size();
    }
  };
  parse_hex_tags("H");
  parse_hex_tags("HEX");

  auto parse_simple = [&](const std::string& tag, int expected, auto& outVec)
  {
    const std::string openTag = "<" + tag;
    const std::string closeTag = "</" + tag + ">";
    size_t pos = 0;
    while (true)
    {
      size_t open = elementSection.find(openTag, pos);
      if (open == std::string::npos) break;
      size_t gt = elementSection.find('>', open);
      if (gt == std::string::npos) break;
      size_t close = elementSection.find(closeTag, gt);
      if (close == std::string::npos) break;

      const std::string header = elementSection.substr(open, gt - open + 1);
      const std::string body = elementSection.substr(gt + 1, close - gt - 1);
      typedef typename std::remove_reference<decltype(outVec)>::type VecT;
      typedef typename VecT::value_type ElemT;
      ElemT elem{};
      elem.id = static_cast<int>(outVec.size());
      const std::string idStr = GetAttribute(header, "ID");
      if (!idStr.empty()) elem.id = std::atoi(idStr.c_str());
      init_array(elem.verts);
      std::istringstream iss(body);
      bool ok = true;
      for (int i = 0; i < expected; ++i)
      {
        if (!(iss >> elem.verts[i]))
        {
          ok = false;
          break;
        }
      }
      if (ok) outVec.push_back(elem);
      pos = close + closeTag.size();
    }
  };

  parse_simple("P", 6, out.prisms);
  parse_simple("PRISM", 6, out.prisms);
  parse_simple("Q", 4, out.quads);
  parse_simple("QUAD", 4, out.quads);
  parse_simple("T", 3, out.tris);
  parse_simple("TRI", 3, out.tris);

  auto find_face_id = [&](const std::vector<int>& vertsWanted, char type)->int
  {
    for (const auto& kv : out.faces)
    {
      const FaceInfo& f = kv.second;
      if (std::toupper(f.type) != std::toupper(type)) continue;
      if (f.verts.size() != vertsWanted.size()) continue;
      bool allMatch = true;
      for (int v : vertsWanted)
      {
        if (std::find(f.verts.begin(), f.verts.end(), v) == f.verts.end())
        {
          allMatch = false;
          break;
        }
      }
      if (allMatch) return kv.first;
    }
    return -1;
  };

  for (auto& pri : out.prisms)
  {
    if (pri.verts[0] < 0) continue;
    init_array(pri.edges);
    const int edgePairs[9][2] = {{0,1},{1,2},{2,0},{3,4},{4,5},{5,3},{0,3},{1,4},{2,5}};
    for (int e = 0; e < 9; ++e)
    {
      int a = pri.verts[edgePairs[e][0]];
      int b = pri.verts[edgePairs[e][1]];
      auto itE = out.edgeLookup.find(EdgeKey(a, b));
      if (itE != out.edgeLookup.end()) pri.edges[e] = itE->second;
    }
    init_array(pri.faces);
    pri.faces[0] = find_face_id({pri.verts[0], pri.verts[1], pri.verts[2]}, 'T');
    pri.faces[1] = find_face_id({pri.verts[3], pri.verts[4], pri.verts[5]}, 'T');
    pri.faces[2] = find_face_id({pri.verts[0], pri.verts[1], pri.verts[4], pri.verts[3]}, 'Q');
    pri.faces[3] = find_face_id({pri.verts[1], pri.verts[2], pri.verts[5], pri.verts[4]}, 'Q');
    pri.faces[4] = find_face_id({pri.verts[2], pri.verts[0], pri.verts[3], pri.verts[5]}, 'Q');
    pri.valid = true;
  }

  // curved
  {
    size_t cStart = xml.find("<CURVED");
    size_t cEnd = xml.find("</CURVED>");
    if (cStart != std::string::npos && cEnd != std::string::npos && cEnd > cStart)
    {
      const std::string block = xml.substr(cStart, cEnd - cStart);
      std::regex edge_re(R"(<E[^>]*EDGEID=\"?(\d+)\"?[^>]*NUMPOINTS=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</E>)", std::regex::icase);
      auto be = std::sregex_iterator(block.begin(), block.end(), edge_re);
      auto en = std::sregex_iterator();
      for (auto it = be; it != en; ++it)
      {
        CurvedEdge ce;
        ce.numPts = static_cast<unsigned int>(std::atoi((*it)[2].str().c_str()));
        const int eid = std::atoi((*it)[1].str().c_str());
        const std::string header = it->str(0);
        const std::string typeAttr = GetAttribute(header, "TYPE");
        if (!typeAttr.empty()) ce.nodeType = ParseNodeType(typeAttr);
        std::istringstream iss((*it)[3].str());
        float3 p{};
        while (iss >> p.x >> p.y >> p.z) ce.pts.push_back(p);
        if (!ce.pts.empty()) out.curvedEdges[eid] = ce;
      }

      std::regex face_re(R"(<F[^>]*FACEID=\"?(\d+)\"?[^>]*NUMPOINTS=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</F>)", std::regex::icase);
      auto bf = std::sregex_iterator(block.begin(), block.end(), face_re);
      for (auto it = bf; it != en; ++it)
      {
        CurvedFace cf;
        cf.numPts = static_cast<unsigned int>(std::atoi((*it)[2].str().c_str()));
        const int fid = std::atoi((*it)[1].str().c_str());
        const std::string header = it->str(0);
        const std::string typeAttr = GetAttribute(header, "TYPE");
        if (!typeAttr.empty()) cf.nodeType = ParseNodeType(typeAttr);
        std::istringstream iss((*it)[3].str());
        float3 p{};
        while (iss >> p.x >> p.y >> p.z) cf.pts.push_back(p);
        if (!cf.pts.empty()) out.curvedFaces[fid] = cf;
      }
    }
  }

  return true;
}
