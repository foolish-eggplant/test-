#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <zlib.h>

#include "fem/ModalBasis.cuh"
#include "raytracer.h"

struct float3
{
  float x, y, z;
};

struct FieldSlice
{
  std::string name;
  int offset{-1};
  int count{0};
  std::array<int,3> modes{0,0,0};
  int hexId{-1};
};

struct AABB
{
  float3 min{1e30f,1e30f,1e30f};
  float3 max{-1e30f,-1e30f,-1e30f};
  void expand(const float3& p)
  {
    min.x = std::min(min.x, p.x); min.y = std::min(min.y, p.y); min.z = std::min(min.z, p.z);
    max.x = std::max(max.x, p.x); max.y = std::max(max.y, p.y); max.z = std::max(max.z, p.z);
  }
};

// ---------- small helpers ----------
std::string ReadFile(const std::string& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string Trim(const std::string& s)
{
  const auto start = s.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return {};
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(start, end - start + 1);
}

std::string GetAttribute(const std::string& tag, const std::string& key)
{
  const auto pos = tag.find(key + "=");
  if (pos == std::string::npos) return {};
  const char quote = tag[pos + key.size() + 1];
  if (quote != '"' && quote != '\'') return {};
  const auto start = pos + key.size() + 2;
  const auto end = tag.find(quote, start);
  if (end == std::string::npos) return {};
  return tag.substr(start, end - start);
}

std::array<int, 3> ParseModes(const std::string& s)
{
  std::array<int, 3> res{0, 0, 0};
  std::vector<int> vals;
  std::string tok;
  for (size_t i = 0; i <= s.size(); ++i)
  {
    char c = (i < s.size()) ? s[i] : ',';
    if (c == ',' || c == ' ' || c == ';')
    {
      if (!tok.empty())
      {
        vals.push_back(std::atoi(tok.c_str()));
        tok.clear();
      }
      continue;
    }
    tok.push_back(c);
  }
  if (vals.size() == 1) res = {vals[0], vals[0], vals[0]};
  else if (vals.size() == 2) res = {vals[0], vals[1], 1};
  else if (vals.size() >= 3) res = {vals[0], vals[1], vals[2]};
  return res;
}

inline int HexCoeffCount(const std::array<int,3>& m)
{
  return std::max(0, m[0]) * std::max(0, m[1]) * std::max(0, m[2]);
}

std::string LargestPayloadBetweenTags(const std::string& text)
{
  size_t pos = 0;
  std::string best;
  while (true)
  {
    size_t open = text.find('>', pos);
    if (open == std::string::npos) break;
    size_t close = text.find('<', open + 1);
    if (close == std::string::npos) break;
    std::string payload = Trim(text.substr(open + 1, close - open - 1));
    if (payload.size() > best.size()) best.swap(payload);
    pos = close + 1;
  }
  return best;
}

std::string ExtractElementsPayload(const std::string& text)
{
  const std::string openTag = "<ELEMENTS";
  const std::string closeTag = "</ELEMENTS>";
  size_t start = text.find(openTag);
  if (start != std::string::npos)
  {
    size_t gt = text.find('>', start);
    size_t end = text.find(closeTag, gt == std::string::npos ? start : gt);
    if (gt != std::string::npos && end != std::string::npos && end > gt)
    {
      return Trim(text.substr(gt + 1, end - gt - 1));
    }
  }
  return LargestPayloadBetweenTags(text);
}

bool ParseAsciiFloats(const std::string& body, std::vector<float>& out)
{
  std::istringstream iss(body);
  float v;
  while (iss >> v) out.push_back(v);
  return !out.empty();
}

std::vector<uint8_t> Base64Decode(const std::string& input)
{
  static const std::string kChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  static std::array<int, 256> kLookup;
  static bool initialized = [] {
    kLookup.fill(-1);
    for (size_t i = 0; i < kChars.size(); ++i)
    {
      kLookup[static_cast<unsigned char>(kChars[i])] = static_cast<int>(i);
    }
    return true;
  }();
  (void)initialized;

  std::vector<uint8_t> out;
  int val = 0;
  int bits = -8;
  for (unsigned char c : input)
  {
    if (std::isspace(c)) continue;
    const int dec = kLookup[c];
    if (dec == -1) break;
    val = (val << 6) + dec;
    bits += 6;
    if (bits >= 0)
    {
      out.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
      bits -= 8;
    }
  }
  return out;
}

bool DecompressZlib(const std::vector<uint8_t>& input, std::vector<uint8_t>& output)
{
  if (input.empty()) return false;
  output.clear();
  auto try_decompress = [&](int windowBits) -> bool
  {
    z_stream strm{};
    strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    strm.avail_in = static_cast<uInt>(input.size());
    if (inflateInit2(&strm, windowBits) != Z_OK) return false;

    std::vector<uint8_t> buffer(128 * 1024);
    output.clear();

    int ret = Z_OK;
    do
    {
      strm.next_out = buffer.data();
      strm.avail_out = static_cast<uInt>(buffer.size());

      ret = inflate(&strm, Z_NO_FLUSH);
      if (ret == Z_STREAM_ERROR || ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR)
      {
        inflateEnd(&strm);
        return false;
      }

      const size_t produced = buffer.size() - strm.avail_out;
      if (produced > 0) output.insert(output.end(), buffer.data(), buffer.data() + produced);
    } while (strm.avail_out == 0);

    inflateEnd(&strm);
    return ret == Z_STREAM_END;
  };

  if (try_decompress(MAX_WBITS)) return true;
  return try_decompress(-MAX_WBITS);
}

std::vector<float> DecodeBinaryCoefficients(const std::vector<uint8_t>& bytes)
{
  std::vector<float> result;
  if (bytes.empty()) return result;

  if (bytes.size() % sizeof(double) == 0)
  {
    const size_t count = bytes.size() / sizeof(double);
    result.resize(count);
    const double* src = reinterpret_cast<const double*>(bytes.data());
    for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(src[i]);
    return result;
  }
  if (bytes.size() % sizeof(float) == 0)
  {
    const size_t count = bytes.size() / sizeof(float);
    result.resize(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
  }
  return result;
}


std::vector<float> GLLNodes(unsigned int n)
{
  std::vector<float> nodes;
  if (n == 0) return nodes;
  if (n == 1) { nodes.push_back(0.0f); return nodes; }
  nodes.resize(n);
  nodes.front() = -1.0f; nodes.back() = 1.0f;
  if (n == 2) return nodes;
  const unsigned int N = n - 1;
  const double pi = 3.14159265358979323846;
  const int maxIter = 50;
  const double tol = 1e-14;
  auto LegendreP = [](int n, double x, double& pnm1)->double {
    if (n == 0) { pnm1 = 0.0; return 1.0; }
    if (n == 1) { pnm1 = 1.0; return x; }
    double p0 = 1.0, p1 = x;
    for (int k = 2; k <= n; ++k)
    {
      double pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
      p0 = p1; p1 = pk;
    }
    pnm1 = p0;
    return p1;
  };
  auto LegendrePDer = [&](int n, double x, double pn, double pnm1)->double {
    const double denom = x * x - 1.0;
    if (std::fabs(denom) < 1e-14) return 0.0;
    return (static_cast<double>(n) * (x * pn - pnm1)) / denom;
  };

  for (unsigned int i = 1; i < n - 1; ++i)
  {
    const unsigned int mirror = n - 1 - i;
    if (i > mirror) { nodes[i] = -nodes[mirror]; continue; }
    double x = -std::cos(pi * static_cast<double>(i) / static_cast<double>(N));
    for (int iter = 0; iter < maxIter; ++iter)
    {
      double pNm1 = 0.0;
      const double pN = LegendreP(static_cast<int>(N), x, pNm1);
      const double pN_der = LegendrePDer(static_cast<int>(N), x, pN, pNm1);
      double pNm2 = 0.0;
      const double pNminus1 = LegendreP(static_cast<int>(N - 1), x, pNm2);
      const double pNminus1_der = LegendrePDer(static_cast<int>(N - 1), x, pNminus1, pNm2);
      const double denom = x * x - 1.0;
      if (std::fabs(denom) < 1e-14) break;
      const double A = x * pN - pNm1;
      const double Ader = pN + x * pN_der - pNminus1_der;
      const double pN_second = static_cast<double>(N) * (Ader * denom - A * (2.0 * x)) / (denom * denom);
      const double dx = pN_der / pN_second;
      x -= dx;
      if (std::fabs(dx) < tol) break;
    }
    nodes[i] = static_cast<float>(x);
    nodes[mirror] = -nodes[i];
  }
  return nodes;
}

std::vector<float> EvenlySpacedNodes(unsigned int n)
{
  std::vector<float> nodes;
  if (n == 0) return nodes;
  if (n == 1) { nodes.push_back(0.0f); return nodes; }
  nodes.resize(n);
  const float step = 2.0f / static_cast<float>(n - 1);
  for (unsigned int i = 0; i < n; ++i) nodes[i] = -1.0f + step * static_cast<float>(i);
  return nodes;
}

std::vector<float> MakeNodes(unsigned int n, bool evenly)
{
  return evenly ? EvenlySpacedNodes(n) : GLLNodes(n);
}

bool InvertSquareMatrixDouble(std::vector<double> mat, int n, std::vector<double>& inv)
{
  inv.assign(n * n, 0.0);
  for (int i = 0; i < n; ++i) inv[i * n + i] = 1.0;
  for (int col = 0; col < n; ++col)
  {
    int pivot = col;
    double maxAbs = std::fabs(mat[col * n + col]);
    for (int r = col + 1; r < n; ++r)
    {
      double v = std::fabs(mat[r * n + col]);
      if (v > maxAbs) { maxAbs = v; pivot = r; }
    }
    if (maxAbs < 1e-13) return false;
    if (pivot != col)
    {
      for (int c = 0; c < n; ++c)
      {
        std::swap(mat[col * n + c], mat[pivot * n + c]);
        std::swap(inv[col * n + c], inv[pivot * n + c]);
      }
    }
    const double invPivot = 1.0 / mat[col * n + col];
    for (int c = 0; c < n; ++c) { mat[col * n + c] *= invPivot; inv[col * n + c] *= invPivot; }
    for (int r = 0; r < n; ++r)
    {
      if (r == col) continue;
      const double factor = mat[r * n + col];
      if (std::fabs(factor) < 1e-13) continue;
      for (int c = 0; c < n; ++c)
      {
        mat[r * n + c] -= factor * mat[col * n + c];
        inv[r * n + c] -= factor * inv[col * n + c];
      }
    }
  }
  return true;
}

std::vector<float> BuildLegendreInterpMatrix(unsigned int nSrc,
                                             unsigned int nTgt,
                                             bool srcEvenly,
                                             bool tgtGLL = true)
{
  const std::vector<float> srcNodes = MakeNodes(nSrc, srcEvenly);
  const std::vector<float> tgtNodes = MakeNodes(nTgt, !tgtGLL);
  std::vector<double> Vsrc(nSrc * nSrc);
  for (unsigned int i = 0; i < nSrc; ++i)
    for (unsigned int j = 0; j < nSrc; ++j)
      Vsrc[i * nSrc + j] = static_cast<double>(zidingyi::ModifiedA(j, srcNodes[i]));
  std::vector<double> VsrcInv;
  if (!InvertSquareMatrixDouble(Vsrc, static_cast<int>(nSrc), VsrcInv)) return {};

  std::vector<float> interp(nTgt * nSrc, 0.0f);
  for (unsigned int i = 0; i < nTgt; ++i)
  {
    for (unsigned int j = 0; j < nSrc; ++j)
    {
      double sum = 0.0;
      for (unsigned int k = 0; k < nSrc; ++k)
      {
        const double basis = static_cast<double>(zidingyi::ModifiedA(k, tgtNodes[i]));
        sum += basis * VsrcInv[k * nSrc + j];
      }
      interp[i * nSrc + j] = static_cast<float>(sum);
    }
  }
  return interp;
}

float CornerDistanceSq(const float3& a, const float3& b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

bool ReorderQuadGrid(const std::vector<float3>& grid,
                     unsigned int dim,
                     const std::array<float3, 4>& desiredCorners,
                     std::vector<float3>& out)
{
  if (grid.size() != static_cast<size_t>(dim * dim) || dim < 2) return false;
  auto fetch = [&](unsigned int i, unsigned int j) -> float3 {
    return grid[j * dim + i];
  };
  const float tol = 1e-5f;
  bool matched = false;
  std::vector<float3> candidate(grid.size());
  int perms[24][4] = {
      {0,1,2,3},{0,1,3,2},{0,2,1,3},{0,2,3,1},{0,3,1,2},{0,3,2,1},
      {1,0,2,3},{1,0,3,2},{1,2,0,3},{1,2,3,0},{1,3,0,2},{1,3,2,0},
      {2,0,1,3},{2,0,3,1},{2,1,0,3},{2,1,3,0},{2,3,0,1},{2,3,1,0},
      {3,0,1,2},{3,0,2,1},{3,1,0,2},{3,1,2,0},{3,2,0,1},{3,2,1,0}};
  for (int swapAxes = 0; swapAxes < 2 && !matched; ++swapAxes)
  {
    for (int flipR = 0; flipR < 2 && !matched; ++flipR)
    {
      for (int flipS = 0; flipS < 2 && !matched; ++flipS)
      {
        auto mapIndex = [&](unsigned int i, unsigned int j) {
          unsigned int ri = swapAxes ? j : i;
          unsigned int sj = swapAxes ? i : j;
          if (flipR) ri = dim - 1 - ri;
          if (flipS) sj = dim - 1 - sj;
          return fetch(ri, sj);
        };
        const float3 c[4] = {mapIndex(0, 0),
                             mapIndex(dim - 1, 0),
                             mapIndex(dim - 1, dim - 1),
                             mapIndex(0, dim - 1)};
        float bestErr = std::numeric_limits<float>::max();
        for (auto& p : perms)
        {
          float err = CornerDistanceSq(c[0], desiredCorners[p[0]]) +
                      CornerDistanceSq(c[1], desiredCorners[p[1]]) +
                      CornerDistanceSq(c[2], desiredCorners[p[2]]) +
                      CornerDistanceSq(c[3], desiredCorners[p[3]]);
          if (err < bestErr) bestErr = err;
        }
        if (bestErr < tol)
        {
          for (unsigned int j = 0; j < dim; ++j)
            for (unsigned int i = 0; i < dim; ++i)
              candidate[j * dim + i] = mapIndex(i, j);
          matched = true;
          break;
        }
      }
    }
  }
  if (matched) out.swap(candidate);
  return matched;
}

std::vector<float3> ResampleQuadFaceLegendre(const std::vector<float3>& src,
                                             unsigned int targetDimR,
                                             unsigned int targetDimS,
                                             const std::array<float3, 4>& expectedCorners)
{
  const size_t srcSize = src.size();
  const unsigned int srcDim = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(srcSize))));
  if (srcDim * srcDim != srcSize || targetDimR == 0 || targetDimS == 0) return {};

  std::vector<float3> oriented = src;
  ReorderQuadGrid(src, srcDim, expectedCorners, oriented);

  const std::vector<float> rowMat = BuildLegendreInterpMatrix(srcDim, targetDimR, true, true);
  const std::vector<float> colMat = BuildLegendreInterpMatrix(srcDim, targetDimS, true, true);
  if (rowMat.empty() || colMat.empty()) return {};

  std::vector<float3> temp(static_cast<size_t>(srcDim) * targetDimR);
  for (unsigned int j = 0; j < srcDim; ++j)
  {
    for (unsigned int i = 0; i < targetDimR; ++i)
    {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (unsigned int k = 0; k < srcDim; ++k)
      {
        const float w = rowMat[i * srcDim + k];
        const float3 p = oriented[j * srcDim + k];
        sx += w * p.x; sy += w * p.y; sz += w * p.z;
      }
      temp[j * targetDimR + i] = float3{static_cast<float>(sx),
                                        static_cast<float>(sy),
                                        static_cast<float>(sz)};
    }
  }

  std::vector<float3> out(static_cast<size_t>(targetDimS) * targetDimR);
  for (unsigned int i = 0; i < targetDimR; ++i)
  {
    for (unsigned int j = 0; j < targetDimS; ++j)
    {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (unsigned int k = 0; k < srcDim; ++k)
      {
        const float w = colMat[j * srcDim + k];
        const float3 p = temp[k * targetDimR + i];
        sx += w * p.x; sy += w * p.y; sz += w * p.z;
      }
      out[j * targetDimR + i] = float3{static_cast<float>(sx),
                                       static_cast<float>(sy),
                                       static_cast<float>(sz)};
    }
  }
  return out;
}

bool TensorModalFit3D_Double(const std::vector<float3>& nodal,
                             const std::vector<float>& rNodes,
                             const std::vector<float>& sNodes,
                             const std::vector<float>& tNodes,
                             std::vector<float>& coeffX,
                             std::vector<float>& coeffY,
                             std::vector<float>& coeffZ)
{
  const int nr = static_cast<int>(rNodes.size());
  const int ns = static_cast<int>(sNodes.size());
  const int nt = static_cast<int>(tNodes.size());
  if (nr == 0 || ns == 0 || nt == 0 ||
      nodal.size() != static_cast<size_t>(nr * ns * nt)) return false;

  std::vector<double> Vr(nr * nr), Vs(ns * ns), Vt(nt * nt);
  std::vector<double> Vr_inv, Vs_inv, Vt_inv;
  for (int i = 0; i < nr; ++i)
    for (int a = 0; a < nr; ++a)
      Vr[i * nr + a] = static_cast<double>(zidingyi::ModifiedA(a, rNodes[i]));
  for (int j = 0; j < ns; ++j)
    for (int b = 0; b < ns; ++b)
      Vs[j * ns + b] = static_cast<double>(zidingyi::ModifiedA(b, sNodes[j]));
  for (int k = 0; k < nt; ++k)
    for (int c = 0; c < nt; ++c)
      Vt[k * nt + c] = static_cast<double>(zidingyi::ModifiedA(c, tNodes[k]));

  if (!InvertSquareMatrixDouble(Vr, nr, Vr_inv) ||
      !InvertSquareMatrixDouble(Vs, ns, Vs_inv) ||
      !InvertSquareMatrixDouble(Vt, nt, Vt_inv)) return false;

  std::vector<double> tempX_r(nr * ns * nt, 0.0), tempY_r(nr * ns * nt, 0.0), tempZ_r(nr * ns * nt, 0.0);
  for (int k = 0; k < nt; ++k)
  {
    for (int j = 0; j < ns; ++j)
    {
      for (int a = 0; a < nr; ++a)
      {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int i = 0; i < nr; ++i)
        {
          const double w = Vr_inv[a * nr + i];
          const float3 v = nodal[(k * ns + j) * nr + i];
          sx += w * v.x; sy += w * v.y; sz += w * v.z;
        }
        const size_t idx = static_cast<size_t>(k * ns + j) * nr + a;
        tempX_r[idx] = sx; tempY_r[idx] = sy; tempZ_r[idx] = sz;
      }
    }
  }

  std::vector<double> tempX_rs(nr * ns * nt, 0.0), tempY_rs(nr * ns * nt, 0.0), tempZ_rs(nr * ns * nt, 0.0);
  for (int k = 0; k < nt; ++k)
  {
    for (int b = 0; b < ns; ++b)
    {
      for (int a = 0; a < nr; ++a)
      {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int j = 0; j < ns; ++j)
        {
          const double w = Vs_inv[b * ns + j];
          const size_t idx = static_cast<size_t>(k * ns + j) * nr + a;
          sx += w * tempX_r[idx];
          sy += w * tempY_r[idx];
          sz += w * tempZ_r[idx];
        }
        const size_t idxOut = static_cast<size_t>(k * ns + b) * nr + a;
        tempX_rs[idxOut] = sx; tempY_rs[idxOut] = sy; tempZ_rs[idxOut] = sz;
      }
    }
  }

  coeffX.assign(static_cast<size_t>(nr) * ns * nt, 0.0f);
  coeffY.assign(static_cast<size_t>(nr) * ns * nt, 0.0f);
  coeffZ.assign(static_cast<size_t>(nr) * ns * nt, 0.0f);
  for (int c = 0; c < nt; ++c)
  {
    for (int b = 0; b < ns; ++b)
    {
      for (int a = 0; a < nr; ++a)
      {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int k = 0; k < nt; ++k)
        {
          const double w = Vt_inv[c * nt + k];
          const size_t idx = static_cast<size_t>(k * ns + b) * nr + a;
          sx += w * tempX_rs[idx];
          sy += w * tempY_rs[idx];
          sz += w * tempZ_rs[idx];
        }
        const size_t idx = (static_cast<size_t>(c) * ns + b) * nr + a;
        coeffX[idx] = static_cast<float>(sx);
        coeffY[idx] = static_cast<float>(sy);
        coeffZ[idx] = static_cast<float>(sz);
      }
    }
  }
  return true;
}

void WritePPM(const std::string& path, const std::vector<float3>& pts, int width = 1024, int height = 768)
{
  if (pts.empty()) return;
  AABB box;
  for (const auto& p : pts) box.expand(p);
  const float dx = box.max.x - box.min.x;
  const float dy = box.max.y - box.min.y;
  const float scale = (dx > dy) ? (width / (dx + 1e-6f)) : (height / (dy + 1e-6f));
  const float cx = 0.5f * (box.min.x + box.max.x);
  const float cy = 0.5f * (box.min.y + box.max.y);

  std::vector<unsigned char> img(static_cast<size_t>(width) * height * 3, 230); // light bg
  auto put = [&](int x, int y, unsigned char r, unsigned char g, unsigned char b) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    size_t idx = (static_cast<size_t>(y) * width + x) * 3;
    img[idx + 0] = r; img[idx + 1] = g; img[idx + 2] = b;
  };

  for (const auto& p : pts)
  {
    int px = static_cast<int>((p.x - cx) * scale + width * 0.5f);
    int py = static_cast<int>((p.y - cy) * scale + height * 0.5f);
    // depth-based shade using z
    const float zNorm = (p.z - box.min.z) / (box.max.z - box.min.z + 1e-6f);
    const unsigned char c = static_cast<unsigned char>(20 + 200 * (1.0f - zNorm));
    for (int dyPix = -1; dyPix <= 1; ++dyPix)
      for (int dxPix = -1; dxPix <= 1; ++dxPix)
        put(px + dxPix, py + dyPix, c, c, c);
  }

  std::ofstream out(path, std::ios::binary);
  out << "P6\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
  std::cout << "Wrote " << path << " (" << width << "x" << height << ")\n";
}

// ---------- loader workflow (hex-only) ----------
bool load_hex_fld(const std::string& xmlPath,
                  const std::string& fldPath,
                  Scene& scene)
{
  std::cout << "[hex_loader] xml=" << xmlPath << " fld=" << fldPath << "\n";

  const std::string xml = ReadFile(xmlPath);
  if (xml.empty())
  {
    std::cerr << "Failed to read xml\n";
    return 1;
  }

  // vertices
  std::vector<float3> verts;
  {
    size_t pos = 0;
    while (true)
    {
      size_t open = xml.find("<V", pos);
      if (open == std::string::npos) break;
      size_t gt = xml.find('>', open);
      size_t close = xml.find("</V>", gt);
      if (gt == std::string::npos || close == std::string::npos) break;
      const std::string header = xml.substr(open, gt - open + 1);
      const std::string body = xml.substr(gt + 1, close - gt - 1);
      int id = static_cast<int>(verts.size());
      const std::string idStr = GetAttribute(header, "ID");
      if (!idStr.empty()) id = std::atoi(idStr.c_str());
      float3 v{0, 0, 0};
      std::istringstream iss(body);
      iss >> v.x >> v.y >> v.z;
      if (id >= static_cast<int>(verts.size())) verts.resize(id + 1);
      verts[id] = v;
      pos = close + 4;
    }
  }
  std::cout << "Vertices: " << verts.size() << "\n";

  // modes
  std::array<int, 3> modes{1, 1, 1};
  {
    const size_t pos = xml.find("NUMMODES");
    if (pos != std::string::npos)
    {
      size_t tagStart = xml.rfind('<', pos);
      size_t tagEnd = xml.find('>', pos);
      if (tagStart != std::string::npos && tagEnd != std::string::npos)
      {
        const std::string tag = xml.substr(tagStart, tagEnd - tagStart + 1);
        const std::string numStr = GetAttribute(tag, "NUMMODES");
        if (!numStr.empty()) modes = ParseModes(numStr);
      }
    }
  }
  std::cout << "Modes (from xml): " << modes[0] << "," << modes[1] << "," << modes[2] << "\n";

  // load fld coefficients (hex-only)
  std::vector<std::string> fieldNames;
  std::vector<float> coeffs;
  {
    const std::string fld = ReadFile(fldPath);
    if (!fld.empty())
    {
      std::regex fields_re(R"(FIELDS=\"([^\"]+)\")", std::regex::icase);
      std::smatch m;
      if (std::regex_search(fld, m, fields_re))
      {
        std::stringstream ss(m[1].str());
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
          tok = Trim(tok);
          if (!tok.empty()) fieldNames.push_back(tok);
        }
      }
      const std::string payload = ExtractElementsPayload(fld);
      if (!payload.empty())
      {
        ParseAsciiFloats(payload, coeffs);
        if (coeffs.empty())
        {
          const std::vector<uint8_t> decoded = Base64Decode(payload);
          std::vector<uint8_t> inflated;
          if (!decoded.empty())
          {
            if (!DecompressZlib(decoded, inflated)) inflated = decoded;
            if (!inflated.empty()) coeffs = DecodeBinaryCoefficients(inflated);
          }
        }
      }
    }
    if (fieldNames.empty()) fieldNames.push_back("u");
    std::cout << "Fields: ";
    for (size_t i = 0; i < fieldNames.size(); ++i)
      std::cout << fieldNames[i] << (i + 1 == fieldNames.size() ? "\n" : ",");
    std::cout << "Coeff count read: " << coeffs.size() << "\n";
  }

  // curved faces (collect all)
  struct CurvedFaceData { unsigned int numPts{0}; std::vector<float3> pts; };
  std::unordered_map<int, CurvedFaceData> curvedFaces;
  {
    size_t cStart = xml.find("<CURVED");
    size_t cEnd = xml.find("</CURVED>");
    if (cStart != std::string::npos && cEnd != std::string::npos && cEnd > cStart)
    {
      const std::string block = xml.substr(cStart, cEnd - cStart);
      std::regex face_re(R"(<F[^>]*FACEID=\"?(\d+)\"?[^>]*NUMPOINTS=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</F>)",
                         std::regex::icase);
      auto b = std::sregex_iterator(block.begin(), block.end(), face_re);
      auto e = std::sregex_iterator();
      for (auto it = b; it != e; ++it)
      {
        CurvedFaceData cf;
        cf.numPts = static_cast<unsigned int>(std::atoi((*it)[2].str().c_str()));
        const int fid = std::atoi((*it)[1].str().c_str());
        std::istringstream iss((*it)[3].str());
        float3 p{};
        while (iss >> p.x >> p.y >> p.z) cf.pts.push_back(p);
        if (!cf.pts.empty()) curvedFaces[fid] = cf;
      }
    }
  }

  // hex connectivity (all)
  struct HexElem { int id{0}; std::array<int,8> v{}; };
  std::vector<HexElem> hexes;
  auto parse_hex = [&](const std::string& tag)
  {
    const std::string openTag = "<" + tag;
    const std::string closeTag = "</" + tag + ">";
    size_t pos = 0;
    while (true)
    {
      size_t open = xml.find(openTag, pos);
      if (open == std::string::npos) break;
      size_t gt = xml.find('>', open);
      size_t close = xml.find(closeTag, gt);
      if (gt == std::string::npos || close == std::string::npos) break;
      const std::string header = xml.substr(open, gt - open + 1);
      const std::string body = xml.substr(gt + 1, close - gt - 1);
      HexElem h{};
      h.v.fill(-1);
      std::istringstream iss(body);
      for (int i = 0; i < 8; ++i) iss >> h.v[i];
      const std::string idStr = GetAttribute(header, "ID");
      if (!idStr.empty()) h.id = std::atoi(idStr.c_str());
      if (h.v[0] >= 0) hexes.push_back(h);
      pos = close + closeTag.size();
    }
  };
  parse_hex("H");
  parse_hex("HEX");
  if (hexes.empty())
  {
    std::cerr << "No hex element found\n";
    return 1;
  }
  std::cout << "Hex count: " << hexes.size() << "\n";

  std::unordered_map<int,size_t> hex_id_to_idx;
  for (size_t i = 0; i < hexes.size(); ++i) hex_id_to_idx[hexes[i].id] = i;

  // composites (only H)
  std::unordered_map<int, std::vector<int>> composites;
  {
    std::regex comp_re(R"(<C\s+ID=\"?(\d+)\"?\s*>\s*([^<]+)\s*</C>)", std::regex::icase);
    auto begin = std::sregex_iterator(xml.begin(), xml.end(), comp_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
    {
      int cid = std::atoi((*it)[1].str().c_str());
      std::string body = (*it)[2].str();
      std::regex entry_re(R"(([Hh])\s*\[\s*([0-9\-\:, ]+)\s*\])");
      auto b2 = std::sregex_iterator(body.begin(), body.end(), entry_re);
      for (auto jt = b2; jt != end; ++jt)
      {
        std::string range = (*jt)[2].str();
        std::stringstream ss(range);
        std::string token;
        while (std::getline(ss, token, ','))
        {
          int lo = 0, hi = 0;
          if (token.find('-') != std::string::npos)
          {
            sscanf(token.c_str(), "%d-%d", &lo, &hi);
          }
          else
          {
            lo = hi = std::atoi(token.c_str());
          }
          for (int v = lo; v <= hi; ++v) composites[cid].push_back(v);
        }
      }
    }
  }

  struct ElemRecord
  {
    int id{-1};
    std::array<int,3> modes{1,1,1};
    std::vector<std::string> fields;
  };
  std::vector<ElemRecord> elem_records;
  {
    std::regex exp_re(R"(<E\s+([^>]*)>)", std::regex::icase);
    auto begin = std::sregex_iterator(xml.begin(), xml.end(), exp_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
    {
      const std::string header = (*it)[1].str();
      const std::string compositeAttr = GetAttribute("<E " + header + ">", "COMPOSITE");
      const std::string numModesPerDir = GetAttribute("<E " + header + ">", "NUMMODESPERDIR");
      const std::string numModesStr = GetAttribute("<E " + header + ">", "NUMMODES");
      const std::string fieldsAttr = GetAttribute("<E " + header + ">", "FIELDS");
      std::array<int,3> emodes = !numModesPerDir.empty() ? ParseModes(numModesPerDir)
                                                         : (!numModesStr.empty() ? ParseModes(numModesStr) : modes);
      std::vector<std::string> efields;
      if (!fieldsAttr.empty())
      {
        std::stringstream ss(fieldsAttr);
        std::string tok;
        while (std::getline(ss, tok, ',')) { tok = Trim(tok); if (!tok.empty()) efields.push_back(tok); }
      }

      std::regex cid_re(R"(C\[\s*([0-9\-]+)\s*\])", std::regex::icase);
      auto bc = std::sregex_iterator(compositeAttr.begin(), compositeAttr.end(), cid_re);
      for (auto ct = bc; ct != end; ++ct)
      {
        std::string token = (*ct)[1].str();
        int lo = 0, hi = 0;
        if (token.find('-') != std::string::npos)
        {
          sscanf(token.c_str(), "%d-%d", &lo, &hi);
        }
        else
        {
          lo = hi = std::atoi(token.c_str());
        }
        for (int cid = lo; cid <= hi; ++cid)
        {
          auto itc = composites.find(cid);
          if (itc != composites.end())
          {
            for (int hid : itc->second)
            {
              ElemRecord rec;
              rec.id = hid;
              rec.modes = emodes;
              rec.fields = efields;
              elem_records.push_back(rec);
            }
          }
        }
      }
    }
  }
  if (elem_records.empty())
  {
    for (const auto& h : hexes)
    {
      ElemRecord rec;
      rec.id = h.id;
      rec.modes = modes;
      rec.fields = fieldNames;
      elem_records.push_back(rec);
    }
  }

  // max geom modes per hex
  std::unordered_map<int, std::array<int,3>> hex_mode_map;
  auto max3 = [](std::array<int,3> a, std::array<int,3> b)->std::array<int,3> {
    return {std::max(a[0], b[0]), std::max(a[1], b[1]), std::max(a[2], b[2])};
  };
  for (const auto& rec : elem_records)
  {
    auto it = hex_mode_map.find(rec.id);
    if (it == hex_mode_map.end()) hex_mode_map[rec.id] = rec.modes;
    else it->second = max3(it->second, rec.modes);
  }

  // field slices per hex following expansion order (hex-only)
  std::vector<std::vector<FieldSlice>> hex_field_slices(hexes.size());
  int coeff_base = 0;
  for (const auto& fname : fieldNames)
  {
    int cursor = coeff_base;
    for (const auto& rec : elem_records)
    {
      const std::vector<std::string>& fList = rec.fields.empty() ? fieldNames : rec.fields;
      if (std::find(fList.begin(), fList.end(), fname) == fList.end()) continue;
      const int cnt = HexCoeffCount(rec.modes);
      if (cnt <= 0) continue;
      FieldSlice fs;
      fs.name = fname;
      fs.offset = cursor;
      fs.count = cnt;
      fs.modes = rec.modes;
      fs.hexId = rec.id;
      auto itH = hex_id_to_idx.find(rec.id);
      if (itH != hex_id_to_idx.end()) hex_field_slices[itH->second].push_back(fs);
      cursor += cnt;
    }
    coeff_base = cursor;
  }

  auto corner = [&](int idx)->float3 {
    if (idx >= 0 && idx < static_cast<int>(verts.size())) return verts[idx];
    return float3{0, 0, 0};
  };

  auto face_corners = [&](int lf)->std::array<float3, 4> {
    switch (lf)
    {
      case 0: return {corner(hex.v[0]), corner(hex.v[1]), corner(hex.v[2]), corner(hex.v[3])};
      case 1: return {corner(hex.v[0]), corner(hex.v[1]), corner(hex.v[5]), corner(hex.v[4])};
      case 2: return {corner(hex.v[1]), corner(hex.v[2]), corner(hex.v[6]), corner(hex.v[5])};
      case 3: return {corner(hex.v[3]), corner(hex.v[2]), corner(hex.v[6]), corner(hex.v[7])};
      case 4: return {corner(hex.v[0]), corner(hex.v[3]), corner(hex.v[7]), corner(hex.v[4])};
      case 5: return {corner(hex.v[4]), corner(hex.v[5]), corner(hex.v[6]), corner(hex.v[7])};
      default: return {corner(0), corner(0), corner(0), corner(0)};
    }
  };

  // target order
  int maxOrder = std::max({modes[0], modes[1], modes[2]});
  for (const auto& kv : curvedFaces)
  {
    const unsigned int n = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(kv.second.numPts))));
    if (n > 1) maxOrder = std::max(maxOrder, static_cast<int>(n));
  }
  if (geomOrder > 0) maxOrder = geomOrder;
  const int nx = maxOrder;
  const int ny = maxOrder;
  const int nz = maxOrder;
  std::cout << "Geom order: " << maxOrder << "\n";

  // field slices (hex-only)
  std::vector<FieldSlice> fieldSlices;
  const int coeffPerHex = modes[0] * modes[1] * modes[2];
  int cursor = 0;
  for (const auto& fname : fieldNames)
  {
    FieldSlice fs;
    fs.name = fname;
    fs.offset = cursor;
    fs.count = coeffPerHex;
    fs.modes = modes;
    fieldSlices.push_back(fs);
    cursor += coeffPerHex;
  }
  if (cursor > static_cast<int>(coeffs.size()))
  {
    std::cout << "Warning: coeffs needed " << cursor << " but only " << coeffs.size() << " available; trailing values assumed 0\n";
  }

  const std::vector<float> rNodes = GLLNodes(nx);
  const std::vector<float> sNodes = GLLNodes(ny);
  const std::vector<float> tNodes = GLLNodes(nz);
  auto idx = [&](int i, int j, int k) { return (static_cast<size_t>(k) * ny + j) * nx + i; };

  // build each hex
  for (size_t hIdx = 0; hIdx < hexes.size(); ++hIdx)
  {
    const HexElem& hex = hexes[hIdx];

    std::array<int,3> baseModes = modes;
    auto itGeom = hex_mode_map.find(hex.id);
    if (itGeom != hex_mode_map.end()) baseModes = itGeom->second;

    int maxOrder = std::max({baseModes[0], baseModes[1], baseModes[2]});
    for (int lf = 0; lf < 6; ++lf)
    {
      auto cfIt = curvedFaces.find(lf);
      if (cfIt != curvedFaces.end() && cfIt->second.numPts > 0)
      {
        const unsigned int n = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(cfIt->second.numPts))));
        if (n > 1) maxOrder = std::max(maxOrder, static_cast<int>(n));
      }
    }
    const int nx = maxOrder;
    const int ny = maxOrder;
    const int nz = maxOrder;
    const std::vector<float> rNodes = GLLNodes(nx);
    const std::vector<float> sNodes = GLLNodes(ny);
    const std::vector<float> tNodes = GLLNodes(nz);
    auto idx3 = [&](int i, int j, int k) { return (static_cast<size_t>(k) * ny + j) * nx + i; };

    // linear base
    std::vector<float3> nodal(static_cast<size_t>(nx) * ny * nz, float3{0, 0, 0});
    auto shapeWeight = [](int idx, float r, float s, float t) {
      const float rSign = (idx & 1) ? 1.0f : -1.0f;
      const float sSign = (idx & 2) ? 1.0f : -1.0f;
      const float tSign = (idx & 4) ? 1.0f : -1.0f;
      return 0.125f * (1.0f + rSign * r) * (1.0f + sSign * s) * (1.0f + tSign * t);
    };
    float3 corners[8];
    for (int i = 0; i < 8; ++i) corners[i] = corner(hex.v[i]);
    for (int k = 0; k < nz; ++k)
      for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
          const float r = rNodes[i], s = sNodes[j], t = tNodes[k];
          float3 p{0, 0, 0};
          for (int c = 0; c < 8; ++c)
          {
            const float w = shapeWeight(c, r, s, t);
            p.x += w * corners[c].x;
            p.y += w * corners[c].y;
            p.z += w * corners[c].z;
          }
          nodal[idx3(i, j, k)] = p;
        }

    // face overrides
    auto blend_face = [&](int lf, const std::vector<float3>& faceGrid)
    {
      if (faceGrid.empty()) return;
      if (lf == 0) // bottom t=-1
      {
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i)
          {
            const float3 target = faceGrid[static_cast<size_t>(j) * nx + i];
            const float3 base = nodal[idx3(i, j, 0)];
            const float3 diff{target.x - base.x, target.y - base.y, target.z - base.z};
            for (int k = 0; k < nz; ++k)
            {
              const float w = 0.5f * (1.0f - tNodes[k]);
              float3& p = nodal[idx3(i, j, k)];
              p.x += diff.x * w; p.y += diff.y * w; p.z += diff.z * w;
            }
          }
      }
      else if (lf == 5) // top t=+1
      {
        for (int j = 0; j < ny; ++j)
          for (int i = 0; i < nx; ++i)
          {
            const float3 target = faceGrid[static_cast<size_t>(j) * nx + i];
            const float3 base = nodal[idx3(i, j, nz - 1)];
            const float3 diff{target.x - base.x, target.y - base.y, target.z - base.z};
            for (int k = 0; k < nz; ++k)
            {
              const float w = 0.5f * (1.0f + tNodes[k]);
              float3& p = nodal[idx3(i, j, k)];
              p.x += diff.x * w; p.y += diff.y * w; p.z += diff.z * w;
            }
          }
      }
    };
    for (int lf = 0; lf < 6; ++lf)
    {
      auto cfIt = curvedFaces.find(lf);
      if (cfIt == curvedFaces.end() || cfIt->second.pts.empty()) continue;
      const std::array<float3,4> fc = face_corners(lf);
      std::vector<float3> faceGrid = ResampleQuadFaceLegendre(cfIt->second.pts,
                                                              static_cast<unsigned int>(nx),
                                                              static_cast<unsigned int>(ny),
                                                              fc);
      if (faceGrid.size() == static_cast<size_t>(nx * ny)) blend_face(lf, faceGrid);
    }

    // modal fitting for geometry
    std::vector<float> gx, gy, gz;
    if (!TensorModalFit3D_Double(nodal, rNodes, sNodes, tNodes, gx, gy, gz)) continue;
    const uint3 geomModes = make_uint3(nx, ny, nz);

    // field slice for this hex
    const std::vector<FieldSlice>& slices = hex_field_slices[hIdx];
    const FieldSlice* fsPtr = nullptr;
    for (const auto& fs : slices)
    {
      if (fs.offset >= 0 && fs.offset + fs.count <= static_cast<int>(coeffs.size()))
      {
        fsPtr = &fs;
        break;
      }
    }
    const float* cptr = nullptr;
    uint3 fm = make_uint3(modes[0], modes[1], modes[2]);
    size_t fcnt = 0;
    if (fsPtr)
    {
      cptr = coeffs.data() + fsPtr->offset;
      fm = make_uint3(fsPtr->modes[0], fsPtr->modes[1], fsPtr->modes[2]);
      fcnt = static_cast<size_t>(fsPtr->count);
    }

    zidingyi::HexElementData h{};
    for (int v = 0; v < 8; ++v) h.vertices[v] = corners[v];
    h.fieldModes = fm;
    scene.add_hex(h,
                  cptr,
                  fm,
                  fcnt,
                  gx.data(),
                  gy.data(),
                  gz.data(),
                  geomModes,
                  gx.size(),
                  &slices);
  }

  return true;
}
