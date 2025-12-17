#include "xml_fld_loader.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <cfloat>
#include <unordered_set>
#include <vector>

#include "xml_fld_common.h"
#include "xml_fld_field_parser.h"
#include "xml_fld_geometry_builder.h"
#include "xml_fld_geometry_parser.h"
#include "high_order.h"

bool load_xml_fld(const std::string& xml_path,
                  const std::string& fld_path,
                  Scene& scene)
{
  std::cout << "XML/FLD loader: mesh=" << xml_path << " fld=" << fld_path << "..." << std::endl;
  const std::string xmlText = ReadFile(xml_path);
  if (xmlText.empty())
  {
    std::cerr << "XML/FLD loader: failed to read xml file." << std::endl;
    return false;
  }

  GeometryData geom;
  if (!ParseGeometry(xmlText, geom))
  {
    std::cerr << "XML/FLD loader: failed to parse geometry block." << std::endl;
    return false;
  }

  FieldData fld = ParseFldFile(ReadFile(fld_path));
  if (fld.fieldNames.empty()) fld.fieldNames.push_back("u");

  auto composites = ParseComposites(xmlText);
  std::vector<ExpansionSpec> expansion_specs = ParseExpansions(xmlText, geom.defaultModes, fld.fieldNames, composites);
  const bool hasExplicitExpansions = !expansion_specs.empty();
  if (expansion_specs.empty())
  {
    ExpansionSpec fallback;
    fallback.modes = geom.defaultModes;
    fallback.fields = fld.fieldNames;
    for (const auto& h : geom.hexes) fallback.elems.push_back({'H', h.id});
    for (const auto& p : geom.prisms) fallback.elems.push_back({'P', p.id});
    for (const auto& q : geom.quads) fallback.elems.push_back({'Q', q.id});
    for (const auto& t : geom.tris) fallback.elems.push_back({'T', t.id});
    expansion_specs.push_back(fallback);
  }

  struct ElemRecord
  {
    char typ{'H'};
    int id{-1};
    uint3 modes{make_uint3(0, 0, 0)};
    std::vector<std::string> fields;
  };
  std::vector<ElemRecord> elem_records;
  auto append_records = [&](const ExpansionSpec& spec)
  {
    for (auto& e : spec.elems)
    {
      ElemRecord rec;
      rec.typ = static_cast<char>(std::toupper(e.first));
      rec.id = e.second;
      rec.modes = spec.modes;
      rec.fields = spec.fields;
      elem_records.push_back(rec);
    }
  };
  for (const auto& spec : expansion_specs) append_records(spec);

  if (hasExplicitExpansions)
  {
    std::unordered_set<int> used_hex, used_prism, used_quad, used_tri;
    for (const auto& rec : elem_records)
    {
      switch (rec.typ)
      {
        case 'H': used_hex.insert(rec.id); break;
        case 'P': used_prism.insert(rec.id); break;
        case 'Q': used_quad.insert(rec.id); break;
        case 'T': used_tri.insert(rec.id); break;
      }
    }
    auto filter_by_set = [](auto& vec, const auto& keep)->void
    {
      vec.erase(std::remove_if(vec.begin(), vec.end(),
                               [&](const auto& e) { return keep.find(e.id) == keep.end(); }),
                vec.end());
    };
    filter_by_set(geom.hexes, used_hex);
    filter_by_set(geom.prisms, used_prism);
    filter_by_set(geom.quads, used_quad);
    filter_by_set(geom.tris, used_tri);
  }

  const uint2 quadModes2 = make_uint2(geom.defaultModes.x, geom.defaultModes.y);
  const uint2 triModes2 = make_uint2(geom.defaultModes.x, geom.defaultModes.y);

  std::unordered_map<int, size_t> hex_id_to_idx, prism_id_to_idx, quad_id_to_idx, tri_id_to_idx;
  for (size_t i = 0; i < geom.hexes.size(); ++i) hex_id_to_idx[geom.hexes[i].id] = i;
  for (size_t i = 0; i < geom.prisms.size(); ++i) prism_id_to_idx[geom.prisms[i].id] = i;
  for (size_t i = 0; i < geom.quads.size(); ++i) quad_id_to_idx[geom.quads[i].id] = i;
  for (size_t i = 0; i < geom.tris.size(); ++i) tri_id_to_idx[geom.tris[i].id] = i;

  auto coeff_count = [&](const ElemRecord& rec)->size_t
  {
    switch (rec.typ)
    {
      case 'H': return HexCoeffCount(rec.modes);
      case 'P': return PrismCoeffCount(rec.modes);
      case 'Q': return QuadCoeffCount(make_uint2(rec.modes.x, rec.modes.y));
      case 'T': return TriCoeffCount(make_uint2(rec.modes.x, rec.modes.y));
      default: return 0;
    }
  };

  std::vector<std::vector<Scene::FieldSlice>> hex_slices(geom.hexes.size());
  std::vector<std::vector<Scene::FieldSlice>> prism_slices(geom.prisms.size());
  std::vector<std::vector<Scene::FieldSlice>> quad_slices(geom.quads.size());
  std::vector<std::vector<Scene::FieldSlice>> tri_slices(geom.tris.size());

  auto max3 = [](uint3 a, uint3 b)->uint3 {
    return make_uint3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
  };
  auto max2 = [](uint2 a, uint2 b)->uint2 {
    return make_uint2(std::max(a.x, b.x), std::max(a.y, b.y));
  };
  std::unordered_map<int, uint3> hex_mode_map;
  std::unordered_map<int, uint3> prism_mode_map;
  std::unordered_map<int, uint2> quad_mode_map;
  std::unordered_map<int, uint2> tri_mode_map;
  for (const auto& rec : elem_records)
  {
    if (rec.typ == 'H')
    {
      auto it = hex_mode_map.find(rec.id);
      if (it == hex_mode_map.end()) hex_mode_map[rec.id] = rec.modes;
      else it->second = max3(it->second, rec.modes);
    }
    else if (rec.typ == 'P')
    {
      auto it = prism_mode_map.find(rec.id);
      if (it == prism_mode_map.end()) prism_mode_map[rec.id] = rec.modes;
      else it->second = max3(it->second, rec.modes);
    }
    else if (rec.typ == 'Q')
    {
      uint2 m2 = make_uint2(rec.modes.x, rec.modes.y);
      auto it = quad_mode_map.find(rec.id);
      if (it == quad_mode_map.end()) quad_mode_map[rec.id] = m2;
      else it->second = max2(it->second, m2);
    }
    else if (rec.typ == 'T')
    {
      uint2 m2 = make_uint2(rec.modes.x, rec.modes.y);
      auto it = tri_mode_map.find(rec.id);
      if (it == tri_mode_map.end()) tri_mode_map[rec.id] = m2;
      else it->second = max2(it->second, m2);
    }
  }

  size_t coeffs_required = 0;
  size_t coeff_base = 0;
  for (const auto& fname : fld.fieldNames)
  {
    size_t cursor = coeff_base;
    bool coeff_stream_exhausted = false;
    size_t needed_at_break = 0;
    for (const auto& rec : elem_records)
    {
      const std::vector<std::string>& fList = rec.fields.empty() ? fld.fieldNames : rec.fields;
      if (std::find(fList.begin(), fList.end(), fname) == fList.end()) continue;
      const size_t cnt = coeff_count(rec);
      if (cnt == 0) continue;
      needed_at_break = cursor + cnt;
      coeffs_required = std::max(coeffs_required, needed_at_break);
      if (cursor + cnt > fld.coeffs.size())
      {
        coeff_stream_exhausted = true;
        break;
      }
      Scene::FieldSlice slice;
      slice.name = fname;
      slice.is3D = (rec.typ == 'H' || rec.typ == 'P');
      slice.modes3 = rec.modes;
      slice.modes2 = make_uint2(rec.modes.x, rec.modes.y);
      slice.count = cnt;
      slice.offset = static_cast<int>(cursor);

      if (rec.typ == 'H')
      {
        auto it = hex_id_to_idx.find(rec.id);
        if (it != hex_id_to_idx.end()) hex_slices[it->second].push_back(slice);
      }
      else if (rec.typ == 'P')
      {
        auto it = prism_id_to_idx.find(rec.id);
        if (it != prism_id_to_idx.end()) prism_slices[it->second].push_back(slice);
      }
      else if (rec.typ == 'Q')
      {
        auto it = quad_id_to_idx.find(rec.id);
        if (it != quad_id_to_idx.end()) quad_slices[it->second].push_back(slice);
      }
      else if (rec.typ == 'T')
      {
        auto it = tri_id_to_idx.find(rec.id);
        if (it != tri_id_to_idx.end()) tri_slices[it->second].push_back(slice);
      }
      cursor += cnt;
    }
    if (coeff_stream_exhausted)
    {
      std::cerr << "XML/FLD loader: coefficient stream exhausted while assigning field '" << fname
                << "' (have " << fld.coeffs.size() << ", need at least " << needed_at_break
                << "); remaining elements will fall back to defaults." << std::endl;
    }
    coeff_base = cursor;
  }

  auto first_slice_ptr = [&](const std::vector<Scene::FieldSlice>& vec)->const Scene::FieldSlice*
  {
    if (vec.empty()) return nullptr;
    return &vec.front();
  };

  auto fetch_vertex = [&](int vid) -> float3 {
    if (vid >= 0 && vid < static_cast<int>(geom.vertices.size())) return geom.vertices[vid];
    return make_float3(0, 0, 0);
  };

  for (size_t i = 0; i < geom.hexes.size(); ++i)
  {
    const Scene::FieldSlice* fs = first_slice_ptr(hex_slices[i]);
    const float* cptr = (fs && fs->offset >= 0 && static_cast<size_t>(fs->offset + fs->count) <= fld.coeffs.size())
                        ? fld.coeffs.data() + fs->offset : nullptr;
    uint3 fm = geom.defaultModes;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes3; fcnt = fs->count; }
    zidingyi::HexElementData h{};
    for (int v = 0; v < 8; ++v) h.vertices[v] = fetch_vertex(geom.hexes[i].verts[v]);
    h.fieldModes = fm;
    h.geomModes = make_uint3(0, 0, 0);
    h.aabbMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    h.aabbMax = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    std::vector<float> gx, gy, gz;
    uint3 geomModes = make_uint3(0, 0, 0);
    uint3 baseGeom = geom.defaultModes;
    auto itBase = hex_mode_map.find(geom.hexes[i].id);
    if (itBase != hex_mode_map.end()) baseGeom = itBase->second;
    BuildHexGeometry(geom.hexes[i],
                     geom.vertices,
                     geom.edges,
                     geom.edgeLookup,
                     geom.curvedEdges,
                     geom.curvedFaces,
                     baseGeom,
                     gx,
                     gy,
                     gz,
                     geomModes);
    // Compute true AABB using high-order geom if available
    h.geomModes = geomModes;
    h.geomCoefficients[0] = gx.empty() ? nullptr : gx.data();
    h.geomCoefficients[1] = gy.empty() ? nullptr : gy.data();
    h.geomCoefficients[2] = gz.empty() ? nullptr : gz.data();
    float3 minC, maxC;
    zidingyi::ComputeHexAabb(h, minC, maxC, 6);
    h.aabbMin = minC;
    h.aabbMax = maxC;
    scene.add_hex(h, cptr, fm, fcnt,
                  gx.empty() ? nullptr : gx.data(),
                  gy.empty() ? nullptr : gy.data(),
                  gz.empty() ? nullptr : gz.data(),
                  geomModes,
                  gx.size(),
                  &hex_slices[i]);
  }

  for (size_t i = 0; i < geom.prisms.size(); ++i)
  {
    const Scene::FieldSlice* fs = first_slice_ptr(prism_slices[i]);
    const float* cptr = (fs && fs->offset >= 0 && static_cast<size_t>(fs->offset + fs->count) <= fld.coeffs.size())
                        ? fld.coeffs.data() + fs->offset : nullptr;
    uint3 fm = geom.defaultModes;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes3; fcnt = fs->count; }
    zidingyi::PrismElementData p{};
    for (int v = 0; v < 6; ++v) p.vertices[v] = fetch_vertex(geom.prisms[i].verts[v]);
    p.fieldModes = fm;
    std::vector<float> gx, gy, gz;
    uint3 geomModes = make_uint3(0, 0, 0);
    uint3 baseGeom = geom.defaultModes;
    auto itBase = prism_mode_map.find(geom.prisms[i].id);
    if (itBase != prism_mode_map.end()) baseGeom = itBase->second;
    BuildPrismGeometry(geom.prisms[i],
                       geom.vertices,
                       geom.edges,
                       geom.edgeLookup,
                       geom.curvedEdges,
                       geom.curvedFaces,
                       baseGeom,
                       gx,
                       gy,
                       gz,
                       geomModes);
    scene.add_prism(p, cptr, fm, fcnt,
                    gx.empty() ? nullptr : gx.data(),
                    gy.empty() ? nullptr : gy.data(),
                    gz.empty() ? nullptr : gz.data(),
                    geomModes,
                    gx.size(),
                    &prism_slices[i]);
  }

  for (size_t i = 0; i < geom.quads.size(); ++i)
  {
    const Scene::FieldSlice* fs = first_slice_ptr(quad_slices[i]);
    const float* cptr = (fs && fs->offset >= 0 && static_cast<size_t>(fs->offset + fs->count) <= fld.coeffs.size())
                        ? fld.coeffs.data() + fs->offset : nullptr;
    uint2 fm = quadModes2;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes2; fcnt = fs->count; }
    zidingyi::QuadElementData q{};
    for (int v = 0; v < 4; ++v) q.vertices[v] = fetch_vertex(geom.quads[i].verts[v]);
    std::vector<float> gx, gy, gz;
    uint2 geomModes = make_uint2(0, 0);
    uint2 baseGeom = quadModes2;
    auto itBase = quad_mode_map.find(geom.quads[i].id);
    if (itBase != quad_mode_map.end()) baseGeom = itBase->second;
    BuildQuadGeometry(geom.quads[i],
                      geom.vertices,
                      geom.edgeLookup,
                      geom.curvedEdges,
                      geom.curvedFaces,
                      baseGeom,
                      gx,
                      gy,
                      gz,
                      geomModes);
    scene.add_quad(q, cptr, fm, fcnt,
                   gx.empty() ? nullptr : gx.data(),
                   gy.empty() ? nullptr : gy.data(),
                   gz.empty() ? nullptr : gz.data(),
                   geomModes,
                   gx.size(),
                   &quad_slices[i]);
  }

  for (size_t i = 0; i < geom.tris.size(); ++i)
  {
    const Scene::FieldSlice* fs = first_slice_ptr(tri_slices[i]);
    const float* cptr = (fs && fs->offset >= 0 && static_cast<size_t>(fs->offset + fs->count) <= fld.coeffs.size())
                        ? fld.coeffs.data() + fs->offset : nullptr;
    uint2 fm = triModes2;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes2; fcnt = fs->count; }
    zidingyi::TriElementData t{};
    for (int v = 0; v < 3; ++v) t.vertices[v] = fetch_vertex(geom.tris[i].verts[v]);
    t.fieldModes = fm;
    std::vector<float> gx, gy, gz;
    uint2 geomModes = make_uint2(0, 0);
    uint2 baseGeom = triModes2;
    auto itBase = tri_mode_map.find(geom.tris[i].id);
    if (itBase != tri_mode_map.end()) baseGeom = itBase->second;
    BuildTriGeometry(geom.tris[i],
                     geom.vertices,
                     geom.edgeLookup,
                     geom.edges,
                     geom.curvedEdges,
                     baseGeom,
                     gx,
                     gy,
                     gz,
                     geomModes);
    scene.add_curved_tri(t, cptr, fm, fcnt,
                         gx.empty() ? nullptr : gx.data(),
                         gy.empty() ? nullptr : gy.data(),
                         gz.empty() ? nullptr : gz.data(),
                         geomModes,
                         gx.size(),
                         &tri_slices[i]);
  }

  const size_t coeffs_used = coeff_base;
  if (!fld.coeffs.empty() && coeffs_used < fld.coeffs.size()) {
    std::cout << "XML/FLD loader: unused coeffs = " << (fld.coeffs.size() - coeffs_used) << std::endl;
  } else if (!fld.coeffs.empty() && coeffs_required > fld.coeffs.size()) {
    std::cout << "XML/FLD loader: coefficients appear insufficient, expected at least " << coeffs_required << " values." << std::endl;
  }

  std::cout << "XML/FLD loader: loaded "
            << scene.hexes.size() << " hexes, "
            << scene.prisms.size() << " prisms, "
            << scene.quads.size() << " quads, "
            << scene.curved_tris.size() << " curved tris." << std::endl;
  return true;
}
