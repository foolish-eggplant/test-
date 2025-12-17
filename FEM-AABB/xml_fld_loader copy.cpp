#include "xml_fld_loader.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <array>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <regex>
#include <limits>
#include <type_traits>
#include "fem/ModalBasis.cuh"

#include <cuda_runtime.h>
#define XML_FLD_HAS_ZLIB 1
#include <zlib.h>

namespace
{
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

std::vector<std::string> ParseFieldList(const std::string& s)
{
  std::vector<std::string> fields;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ','))
  {
    std::string name = Trim(tok);
    if (!name.empty()) fields.push_back(name);
  }
  return fields;
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

uint3 ParseModes(const std::string& s)
{
  uint3 result = make_uint3(0, 0, 0);
  std::vector<int> vals;
  std::string token;
  for (size_t i = 0; i <= s.size(); ++i)
  {
    const char c = (i < s.size() ? s[i] : ',');
    if (c == ',' || c == ' ' || c == ';')
    {
      if (!token.empty())
      {
        vals.push_back(std::atoi(token.c_str()));
        token.clear();
      }
      continue;
    }
    token.push_back(c);
  }

  if (vals.empty()) return result;
  if (vals.size() == 1)
  {
    result.x = result.y = result.z = static_cast<unsigned int>(vals[0]);
  }
  else if (vals.size() == 2)
  {
    result.x = static_cast<unsigned int>(vals[0]);
    result.y = static_cast<unsigned int>(vals[1]);
    result.z = 1;
  }
  else
  {
    result.x = static_cast<unsigned int>(vals[0]);
    result.y = static_cast<unsigned int>(vals[1]);
    result.z = static_cast<unsigned int>(vals[2]);
  }
  return result;
}

size_t HexCoeffCount(const uint3& modes)
{
  return static_cast<size_t>(modes.x) * modes.y * modes.z;
}

size_t QuadCoeffCount(const uint2& modes)
{
  return static_cast<size_t>(modes.x) * modes.y;
}

size_t TriCoeffCount(const uint2& modes)
{
  size_t count = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const unsigned int maxJ = (modes.y > i) ? (modes.y - i) : 0;
    count += maxJ;
  }
  return count;
}

size_t PrismCoeffCount(const uint3& modes)
{
  size_t count = 0;
  for (unsigned int i = 0; i < modes.x; ++i)
  {
    const unsigned int maxK = (modes.z > i) ? (modes.z - i) : 0;
    count += static_cast<size_t>(modes.y) * maxK;
  }
  return count;
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

bool ParseAsciiFloats(const std::string& body, std::vector<float>& out)
{
  std::istringstream iss(body);
  float v;
  while (iss >> v) out.push_back(v);
  return !out.empty();
}

std::vector<uint8_t> Base64Decode(const std::string& input)
{
  static const int8_t kDec[256] = {
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
      52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
      -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
      15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
      -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
      41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};

  std::vector<uint8_t> out;
  int val = 0;
  int bits = -8;
  for (unsigned char c : input)
  {
    if (std::isspace(c)) continue;
    const int8_t dec = kDec[c];
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
#if !XML_FLD_HAS_ZLIB
  return false;
#else
  z_stream strm{};
  if (inflateInit(&strm) != Z_OK) return false;

  strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
  strm.avail_in = static_cast<uInt>(input.size());

  const size_t kChunk = 1 << 14;
  std::vector<uint8_t> buffer(kChunk);
  int ret = Z_OK;
  while (ret == Z_OK)
  {
    strm.next_out = buffer.data();
    strm.avail_out = static_cast<uInt>(buffer.size());
    ret = inflate(&strm, Z_NO_FLUSH);
    const size_t produced = buffer.size() - strm.avail_out;
    output.insert(output.end(), buffer.data(), buffer.data() + produced);
  }

  inflateEnd(&strm);
  return !output.empty();
#endif
}

std::vector<float> DecodeBinaryCoefficients(const std::vector<uint8_t>& bytes)
{
  std::vector<float> result;
  if (bytes.empty()) return result;

  if (bytes.size() % sizeof(float) == 0)
  {
    const size_t count = bytes.size() / sizeof(float);
    result.resize(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
  }
  else if (bytes.size() % sizeof(double) == 0)
  {
    const size_t count = bytes.size() / sizeof(double);
    result.resize(count);
    const double* src = reinterpret_cast<const double*>(bytes.data());
    for (size_t i = 0; i < count; ++i) result[i] = static_cast<float>(src[i]);
  }
  return result;
}

double LegendreP(int n, double x, double& pnm1)
{
  if (n == 0)
  {
    pnm1 = 0.0;
    return 1.0;
  }
  if (n == 1)
  {
    pnm1 = 1.0;
    return x;
  }
  double p0 = 1.0;
  double p1 = x;
  for (int k = 2; k <= n; ++k)
  {
    const double pk = ((2.0 * static_cast<double>(k) - 1.0) * x * p1 -
                       (static_cast<double>(k) - 1.0) * p0) /
                      static_cast<double>(k);
    p0 = p1;
    p1 = pk;
  }
  pnm1 = p0;
  return p1;
}

double LegendrePDer(int n, double x, double pn, double pnm1)
{
  const double denom = 1.0 - x * x;
  if (std::fabs(denom) < 1e-14) return 0.0;
  return (static_cast<double>(n) * (pnm1 - x * pn)) / denom;
}

double LegendrePSecond(int n, double x, double pn, double pnp)
{
  const double denom = 1.0 - x * x;
  if (std::fabs(denom) < 1e-14) return 0.0;
  return (2.0 * x * pnp - static_cast<double>(n) * (static_cast<double>(n) + 1.0) * pn) / denom;
}

std::vector<float> GLLNodes(unsigned int n)
{
  std::vector<float> nodes;
  if (n == 0) return nodes;
  if (n == 1)
  {
    nodes.push_back(0.0f);
    return nodes;
  }
  nodes.resize(n);
  nodes.front() = -1.0f;
  nodes.back() = 1.0f;
  if (n == 2) return nodes;

  const unsigned int N = n - 1; // Legendre polynomial order
  const double pi = 3.14159265358979323846;
  const int maxIter = 50;
  const double tol = 1e-14;

  for (unsigned int i = 1; i < n - 1; ++i)
  {
    const unsigned int mirror = n - 1 - i;
    if (i > mirror)
    {
      nodes[i] = -nodes[mirror];
      continue;
    }
    double x = -std::cos(pi * static_cast<double>(i) / static_cast<double>(N));
    for (int iter = 0; iter < maxIter; ++iter)
    {
      double pnm1 = 0.0;
      const double pn = LegendreP(static_cast<int>(N), x, pnm1);
      const double pnp = LegendrePDer(static_cast<int>(N), x, pn, pnm1);
      const double pnp2 = LegendrePSecond(static_cast<int>(N), x, pn, pnp);
      if (std::fabs(pnp2) < 1e-16) break;
      const double dx = pnp / pnp2;
      x -= dx;
      if (std::fabs(dx) < tol) break;
    }
    nodes[i] = static_cast<float>(x);
    nodes[mirror] = -nodes[i];
  }
  return nodes;
}

bool InvertSquareMatrix(std::vector<float> mat, int n, std::vector<float>& inv)
{
  inv.assign(n * n, 0.0f);
  for (int i = 0; i < n; ++i) inv[i * n + i] = 1.0f;

  for (int col = 0; col < n; ++col)
  {
    int pivot = col;
    float maxAbs = std::fabs(mat[col * n + col]);
    for (int r = col + 1; r < n; ++r)
    {
      float v = std::fabs(mat[r * n + col]);
      if (v > maxAbs)
      {
        maxAbs = v;
        pivot = r;
      }
    }
    if (maxAbs < 1e-10f) return false;
    if (pivot != col)
    {
      for (int c = 0; c < n; ++c)
      {
        std::swap(mat[col * n + c], mat[pivot * n + c]);
        std::swap(inv[col * n + c], inv[pivot * n + c]);
      }
    }
    const float invPivot = 1.0f / mat[col * n + col];
    for (int c = 0; c < n; ++c)
    {
      mat[col * n + c] *= invPivot;
      inv[col * n + c] *= invPivot;
    }
    for (int r = 0; r < n; ++r)
    {
      if (r == col) continue;
      const float factor = mat[r * n + col];
      if (std::fabs(factor) < 1e-10f) continue;
      for (int c = 0; c < n; ++c)
      {
        mat[r * n + c] -= factor * mat[col * n + c];
        inv[r * n + c] -= factor * inv[col * n + c];
      }
    }
  }
  return true;
}

inline float EvalModal1D(const std::vector<float>& coeffs, float r)
{
  float val = 0.0f;
  for (size_t i = 0; i < coeffs.size(); ++i)
  {
    val += coeffs[i] * zidingyi::ModifiedA(static_cast<unsigned int>(i), r);
  }
  return val;
}

inline float EvalModal2D(const std::vector<float>& coeffs, uint2 modes, float r, float s)
{
  float val = 0.0f;
  size_t idx = 0;
  for (unsigned int j = 0; j < modes.y; ++j)
  {
    const float sj = zidingyi::ModifiedA(j, s);
    for (unsigned int i = 0; i < modes.x; ++i)
    {
      const float ri = zidingyi::ModifiedA(i, r);
      val += coeffs[idx++] * ri * sj;
    }
  }
  return val;
}

inline float EvalModal3D(const std::vector<float>& coeffs, uint3 modes, float r, float s, float t)
{
  float val = 0.0f;
  size_t idx = 0;
  for (unsigned int k = 0; k < modes.z; ++k)
  {
    const float tk = zidingyi::ModifiedA(k, t);
    for (unsigned int j = 0; j < modes.y; ++j)
    {
      const float sj = zidingyi::ModifiedA(j, s);
      for (unsigned int i = 0; i < modes.x; ++i)
      {
        const float ri = zidingyi::ModifiedA(i, r);
        val += coeffs[idx++] * ri * sj * tk;
      }
    }
  }
  return val;
}

bool TensorModalFit2D(const std::vector<float3>& nodal,
                      const std::vector<float>& rNodes,
                      const std::vector<float>& sNodes,
                      std::vector<float>& coeffX,
                      std::vector<float>& coeffY,
                      std::vector<float>& coeffZ)
{
  const int nr = static_cast<int>(rNodes.size());
  const int ns = static_cast<int>(sNodes.size());
  if (nr == 0 || ns == 0 || nodal.size() != static_cast<size_t>(nr * ns)) return false;

  std::vector<float> Vr(nr * nr), Vs(ns * ns), Vr_inv, Vs_inv;
  for (int i = 0; i < nr; ++i)
    for (int a = 0; a < nr; ++a)
      Vr[i * nr + a] = zidingyi::ModifiedA(a, rNodes[i]);
  for (int j = 0; j < ns; ++j)
    for (int b = 0; b < ns; ++b)
      Vs[j * ns + b] = zidingyi::ModifiedA(b, sNodes[j]);
  if (!InvertSquareMatrix(Vr, nr, Vr_inv) || !InvertSquareMatrix(Vs, ns, Vs_inv)) return false;

  std::vector<float3> temp(nr * ns);
  for (int j = 0; j < ns; ++j)
  {
    for (int a = 0; a < nr; ++a)
    {
      float3 sum = make_float3(0, 0, 0);
      for (int i = 0; i < nr; ++i)
      {
        const float w = Vr_inv[a * nr + i];
        const float3 v = nodal[j * nr + i];
        sum.x += w * v.x;
        sum.y += w * v.y;
        sum.z += w * v.z;
      }
      temp[j * nr + a] = sum;
    }
  }

  coeffX.assign(nr * ns, 0.0f);
  coeffY.assign(nr * ns, 0.0f);
  coeffZ.assign(nr * ns, 0.0f);
  for (int b = 0; b < ns; ++b)
  {
    for (int a = 0; a < nr; ++a)
    {
      float sx = 0.0f, sy = 0.0f, sz = 0.0f;
      for (int j = 0; j < ns; ++j)
      {
        const float w = Vs_inv[b * ns + j];
        const float3 v = temp[j * nr + a];
        sx += w * v.x;
        sy += w * v.y;
        sz += w * v.z;
      }
      const size_t idx = static_cast<size_t>(b) * nr + a;
      coeffX[idx] = sx;
      coeffY[idx] = sy;
      coeffZ[idx] = sz;
    }
  }
  return true;
}

bool TensorModalFit3D(const std::vector<float3>& nodal,
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

  std::vector<float> Vr(nr * nr), Vs(ns * ns), Vt(nt * nt);
  std::vector<float> Vr_inv, Vs_inv, Vt_inv;
  for (int i = 0; i < nr; ++i)
    for (int a = 0; a < nr; ++a)
      Vr[i * nr + a] = zidingyi::ModifiedA(a, rNodes[i]);
  for (int j = 0; j < ns; ++j)
    for (int b = 0; b < ns; ++b)
      Vs[j * ns + b] = zidingyi::ModifiedA(b, sNodes[j]);
  for (int k = 0; k < nt; ++k)
    for (int c = 0; c < nt; ++c)
      Vt[k * nt + c] = zidingyi::ModifiedA(c, tNodes[k]);

  if (!InvertSquareMatrix(Vr, nr, Vr_inv) ||
      !InvertSquareMatrix(Vs, ns, Vs_inv) ||
      !InvertSquareMatrix(Vt, nt, Vt_inv)) return false;

  std::vector<float3> temp_r(nr * ns * nt);
  for (int k = 0; k < nt; ++k)
  {
    for (int j = 0; j < ns; ++j)
    {
      for (int a = 0; a < nr; ++a)
      {
        float3 sum = make_float3(0, 0, 0);
        for (int i = 0; i < nr; ++i)
        {
          const float w = Vr_inv[a * nr + i];
          const float3 v = nodal[(k * ns + j) * nr + i];
          sum.x += w * v.x;
          sum.y += w * v.y;
          sum.z += w * v.z;
        }
        temp_r[(k * ns + j) * nr + a] = sum;
      }
    }
  }

  std::vector<float3> temp_rs(nr * ns * nt);
  for (int k = 0; k < nt; ++k)
  {
    for (int b = 0; b < ns; ++b)
    {
      for (int a = 0; a < nr; ++a)
      {
        float3 sum = make_float3(0, 0, 0);
        for (int j = 0; j < ns; ++j)
        {
          const float w = Vs_inv[b * ns + j];
          const float3 v = temp_r[(k * ns + j) * nr + a];
          sum.x += w * v.x;
          sum.y += w * v.y;
          sum.z += w * v.z;
        }
        temp_rs[(k * ns + b) * nr + a] = sum;
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
        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int k = 0; k < nt; ++k)
        {
          const float w = Vt_inv[c * nt + k];
          const float3 v = temp_rs[(k * ns + b) * nr + a];
          sx += w * v.x;
          sy += w * v.y;
          sz += w * v.z;
        }
        const size_t idx = (static_cast<size_t>(c) * ns + b) * nr + a;
        coeffX[idx] = sx;
        coeffY[idx] = sy;
        coeffZ[idx] = sz;
      }
    }
  }
  return true;
}

bool SolveTriModalCoeffs(const std::vector<float3>& nodal,
                         const std::vector<float2>& rsNodes,
                         uint2 modes,
                         std::vector<float>& coeffX,
                         std::vector<float>& coeffY,
                         std::vector<float>& coeffZ)
{
  const size_t nNodes = rsNodes.size();
  const size_t nModes = TriCoeffCount(modes);
  if (nNodes != nModes || nodal.size() != nNodes) return false;

  std::vector<float> V(nModes * nModes);
  size_t row = 0;
  for (size_t p = 0; p < nNodes; ++p)
  {
    const float r = rsNodes[p].x;
    const float s = rsNodes[p].y;
    size_t col = 0;
    for (unsigned int i = 0; i < modes.x; ++i)
    {
      const float ai = zidingyi::ModifiedA(i, r);
      const unsigned int maxJ = (modes.y > i) ? (modes.y - i) : 0;
      for (unsigned int j = 0; j < maxJ; ++j)
      {
        const float bj = zidingyi::ModifiedB(i, j, s);
        V[row * nModes + col] = ai * bj;
        ++col;
      }
    }
    ++row;
  }
  std::vector<float> V_inv;
  if (!InvertSquareMatrix(V, static_cast<int>(nModes), V_inv)) return false;

  coeffX.assign(nModes, 0.0f);
  coeffY.assign(nModes, 0.0f);
  coeffZ.assign(nModes, 0.0f);
  for (size_t m = 0; m < nModes; ++m)
  {
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (size_t p = 0; p < nNodes; ++p)
    {
      const float w = V_inv[m * nModes + p];
      const float3 v = nodal[p];
      sx += w * v.x;
      sy += w * v.y;
      sz += w * v.z;
    }
    coeffX[m] = sx;
    coeffY[m] = sy;
    coeffZ[m] = sz;
  }
  return true;
}

std::vector<float3> ResampleEdgePoints(const std::vector<float3>& src,
                                       unsigned int targetCount,
                                       bool reverseDirection = false)
{
  if (src.empty() || targetCount == 0) return {};
  const unsigned int nSrc = static_cast<unsigned int>(src.size());
  const std::vector<float> rSrc = GLLNodes(nSrc);
  std::vector<float> Vr(nSrc * nSrc), Vr_inv;
  for (unsigned int i = 0; i < nSrc; ++i)
    for (unsigned int a = 0; a < nSrc; ++a)
      Vr[i * nSrc + a] = zidingyi::ModifiedA(a, rSrc[i]);
  if (!InvertSquareMatrix(Vr, static_cast<int>(nSrc), Vr_inv)) return {};

  std::vector<float> coeffX(nSrc, 0.0f), coeffY(nSrc, 0.0f), coeffZ(nSrc, 0.0f);
  for (unsigned int a = 0; a < nSrc; ++a)
  {
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (unsigned int i = 0; i < nSrc; ++i)
    {
      const float w = Vr_inv[a * nSrc + i];
      const float3 v = src[i];
      sx += w * v.x;
      sy += w * v.y;
      sz += w * v.z;
    }
    coeffX[a] = sx;
    coeffY[a] = sy;
    coeffZ[a] = sz;
  }

  const std::vector<float> rT = GLLNodes(targetCount);
  std::vector<float3> out(targetCount);
  for (unsigned int i = 0; i < targetCount; ++i)
  {
    const float r = reverseDirection ? rT[targetCount - 1 - i] : rT[i];
    out[i] = make_float3(EvalModal1D(coeffX, r),
                         EvalModal1D(coeffY, r),
                         EvalModal1D(coeffZ, r));
  }
  return out;
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
  const float tol = 1e-6f;
  bool matched = false;
  std::vector<float3> candidate(grid.size());
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
        const float3 c0 = mapIndex(0, 0);
        const float3 c1 = mapIndex(dim - 1, 0);
        const float3 c2 = mapIndex(dim - 1, dim - 1);
        const float3 c3 = mapIndex(0, dim - 1);
        if (CornerDistanceSq(c0, desiredCorners[0]) < tol &&
            CornerDistanceSq(c1, desiredCorners[1]) < tol &&
            CornerDistanceSq(c2, desiredCorners[2]) < tol &&
            CornerDistanceSq(c3, desiredCorners[3]) < tol)
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
  if (matched)
  {
    out.swap(candidate);
  }
  return matched;
}

bool ReorderTriGrid(const std::vector<float3>& grid,
                    unsigned int order,
                    const std::array<float3, 3>& desiredCorners,
                    std::vector<float3>& out)
{
  const size_t expectedSize = static_cast<size_t>(order) * (order + 1) / 2;
  if (grid.size() != expectedSize || order < 2) return false;
  auto idx = [&](unsigned int i, unsigned int j) -> size_t {
    // rows of length order, order-1, ...; j from 0..order-1
    const unsigned int rowStart = j * order - (j * (j - 1)) / 2;
    return static_cast<size_t>(rowStart + i);
  };

  const float tol = 1e-6f;
  const float3 c0 = grid[idx(0, 0)];
  const float3 c1 = grid[idx(order - 1, 0)];
  const float3 c2 = grid[idx(0, order - 1)];

  int bestPerm[3] = {0, 1, 2};
  float bestErr = std::numeric_limits<float>::max();
  int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
  const float3 corners[3] = {c0, c1, c2};
  for (auto& perm : perms)
  {
    float err = CornerDistanceSq(corners[perm[0]], desiredCorners[0]) +
                CornerDistanceSq(corners[perm[1]], desiredCorners[1]) +
                CornerDistanceSq(corners[perm[2]], desiredCorners[2]);
    if (err < bestErr)
    {
      bestErr = err;
      bestPerm[0] = perm[0];
      bestPerm[1] = perm[1];
      bestPerm[2] = perm[2];
    }
  }
  if (bestErr > tol) return false;

  out.resize(grid.size());
  for (unsigned int j = 0; j < order; ++j)
  {
    for (unsigned int i = 0; i + j < order; ++i)
    {
      const float u = (order == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(order - 1);
      const float v = (order == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(order - 1);
      const float w = std::max(0.0f, 1.0f - u - v);
      float baryOrig[3] = {0.0f, 0.0f, 0.0f};
      baryOrig[bestPerm[0]] = u;
      baryOrig[bestPerm[1]] = v;
      baryOrig[bestPerm[2]] = w;
      unsigned int iOrig = static_cast<unsigned int>(std::round(baryOrig[0] * (order - 1)));
      unsigned int jOrig = static_cast<unsigned int>(std::round(baryOrig[1] * (order - 1)));
      if (iOrig + jOrig >= order)
      {
        const unsigned int excess = iOrig + jOrig - (order - 1);
        if (iOrig >= excess) iOrig -= excess;
        else if (jOrig >= excess) jOrig -= excess;
        else { continue; }
      }
      const size_t srcIdx = idx(iOrig, jOrig);
      const size_t dstIdx = idx(i, j);
      if (srcIdx < grid.size() && dstIdx < out.size()) out[dstIdx] = grid[srcIdx];
    }
  }
  return true;
}

std::vector<float3> ResampleQuadFace(const std::vector<float3>& src,
                                     unsigned int targetDimR,
                                     unsigned int targetDimS,
                                     const std::array<float3, 4>& expectedCorners)
{
  const size_t srcSize = src.size();
  const unsigned int srcDim = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(srcSize))));
  if (srcDim * srcDim != srcSize || targetDimR == 0 || targetDimS == 0) return {};
  std::vector<float3> oriented = src;
  ReorderQuadGrid(src, srcDim, expectedCorners, oriented);

  const std::vector<float> rSrc = GLLNodes(srcDim);
  const std::vector<float> sSrc = GLLNodes(srcDim);
  std::vector<float> coeffX, coeffY, coeffZ;
  if (!TensorModalFit2D(oriented, rSrc, sSrc, coeffX, coeffY, coeffZ)) return {};

  const std::vector<float> rT = GLLNodes(targetDimR);
  const std::vector<float> sT = GLLNodes(targetDimS);
  const uint2 modes = make_uint2(srcDim, srcDim);
  std::vector<float3> out(static_cast<size_t>(targetDimR) * targetDimS);
  for (unsigned int j = 0; j < targetDimS; ++j)
  {
    for (unsigned int i = 0; i < targetDimR; ++i)
    {
      const size_t idx = static_cast<size_t>(j) * targetDimR + i;
      out[idx] = make_float3(
        EvalModal2D(coeffX, modes, rT[i], sT[j]),
        EvalModal2D(coeffY, modes, rT[i], sT[j]),
        EvalModal2D(coeffZ, modes, rT[i], sT[j]));
    }
  }
  return out;
}

std::vector<float3> ResampleTriFace(const std::vector<float3>& src,
                                    unsigned int targetOrder,
                                    const std::array<float3, 3>& expectedCorners)
{
  // Order = number of points along each edge (modes)
  const size_t expectedSrc = static_cast<size_t>(targetOrder) * (targetOrder + 1) / 2;
  if (src.size() < expectedSrc || targetOrder < 2) return {};

  // Build nodes for the source (assume evenly spaced)
  const unsigned int srcOrder = static_cast<unsigned int>(std::round((std::sqrt(8.0 * src.size() + 1.0) - 1.0) * 0.5));
  if (srcOrder * (srcOrder + 1) / 2 != src.size()) return {};

  std::vector<float2> nodesSrc;
  nodesSrc.reserve(src.size());
  for (unsigned int j = 0; j < srcOrder; ++j)
  {
    for (unsigned int i = 0; i + j < srcOrder; ++i)
    {
      const float u = static_cast<float>(i) / static_cast<float>(srcOrder - 1);
      const float v = static_cast<float>(j) / static_cast<float>(srcOrder - 1);
      const float r = 2.0f * u - 1.0f;
      const float s = 2.0f * v - 1.0f;
      nodesSrc.push_back(make_float2(r, s));
    }
  }

  std::vector<float3> oriented = src;
  ReorderTriGrid(src, srcOrder, expectedCorners, oriented);

  std::vector<float> coeffX, coeffY, coeffZ;
  if (!SolveTriModalCoeffs(oriented, nodesSrc, make_uint2(srcOrder, srcOrder), coeffX, coeffY, coeffZ)) return {};

  const unsigned int tgt = targetOrder;
  std::vector<float2> nodesT;
  nodesT.reserve(tgt * (tgt + 1) / 2);
  for (unsigned int j = 0; j < tgt; ++j)
  {
    for (unsigned int i = 0; i + j < tgt; ++i)
    {
      const float u = static_cast<float>(i) / static_cast<float>(tgt - 1);
      const float v = static_cast<float>(j) / static_cast<float>(tgt - 1);
      const float r = 2.0f * u - 1.0f;
      const float s = 2.0f * v - 1.0f;
      nodesT.push_back(make_float2(r, s));
    }
  }

  std::vector<float3> out(nodesT.size());
  const uint2 modes = make_uint2(srcOrder, srcOrder);
  for (size_t idx = 0; idx < nodesT.size(); ++idx)
  {
    const float r = nodesT[idx].x;
    const float s = nodesT[idx].y;
    out[idx] = make_float3(
      EvalModal2D(coeffX, modes, r, s),
      EvalModal2D(coeffY, modes, r, s),
      EvalModal2D(coeffZ, modes, r, s));
  }
  return out;
}

struct PrismNodeKeyHasher
{
  std::size_t operator()(const uint64_t& k) const noexcept { return std::hash<uint64_t>{}(k); }
};

inline uint64_t PrismNodeKey(unsigned int i, unsigned int j, unsigned int k)
{
  return (static_cast<uint64_t>(i) << 42) | (static_cast<uint64_t>(j) << 21) | static_cast<uint64_t>(k);
}

inline float EvaluatePrismBasis(unsigned int i, unsigned int j, unsigned int k,
                                const float3& rst)
{
  const float ri = zidingyi::ModifiedA(i, rst.x);
  const float sj = zidingyi::ModifiedA(j, rst.y);
  const float tk = zidingyi::ModifiedB(i, k, rst.z);
  return ri * sj * tk;
}

inline float3 ReferenceToWorldPrismLinear(const std::array<float3, 6>& v,
                                          const float3& rst)
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

  // --- vertices ---
  std::vector<float3> vertices;
  {
    size_t pos = 0;
    while (true)
    {
      size_t open = xmlText.find("<V", pos);
      if (open == std::string::npos) break;
      size_t gt = xmlText.find('>', open);
      if (gt == std::string::npos) break;
      size_t close = xmlText.find("</V>", gt);
      if (close == std::string::npos) break;

      const std::string tagHeader = xmlText.substr(open, gt - open + 1);
      const std::string body = xmlText.substr(gt + 1, close - gt - 1);
      int id = static_cast<int>(vertices.size());
      const std::string idStr = GetAttribute(tagHeader, "ID");
      if (!idStr.empty()) id = std::atoi(idStr.c_str());
      float3 v = make_float3(0, 0, 0);
      std::istringstream iss(body);
      iss >> v.x >> v.y >> v.z;
      if (id >= static_cast<int>(vertices.size())) vertices.resize(id + 1);
      vertices[id] = v;
      pos = close + 4;
    }
  }

  if (vertices.empty())
  {
    std::cerr << "XML/FLD loader: no <V> vertices found." << std::endl;
    return false;
  }

  // 仅在 <ELEMENT> 区块内寻找单元定义，避免将 <FACE> 下的 Q/T 误当作元素
  const size_t elemBlockStart = xmlText.find("<ELEMENT");
  const size_t elemBlockEnd = xmlText.find("</ELEMENT>");
  const std::string elementSection = (elemBlockStart != std::string::npos &&
                                      elemBlockEnd != std::string::npos &&
                                      elemBlockEnd > elemBlockStart)
                                       ? xmlText.substr(elemBlockStart, elemBlockEnd - elemBlockStart)
                                       : xmlText;

  // --- modes (expansion order) ---
  uint3 modes = make_uint3(0, 0, 0);
  {
    const size_t expPos = xmlText.find("NUMMODES");
    if (expPos != std::string::npos)
    {
      size_t tagStart = xmlText.rfind('<', expPos);
      size_t tagEnd = xmlText.find('>', expPos);
      if (tagStart != std::string::npos && tagEnd != std::string::npos && tagEnd > tagStart)
      {
        const std::string tag = xmlText.substr(tagStart, tagEnd - tagStart + 1);
        const std::string numModesStr = GetAttribute(tag, "NUMMODES");
        if (!numModesStr.empty()) modes = ParseModes(numModesStr);
      }
    }
    if (modes.x == 0) modes = make_uint3(1, 1, 1);
  }

// --- topology: edges, faces, elements ---
  struct EdgeInfo { int v0{-1}; int v1{-1}; };
  std::unordered_map<int, EdgeInfo> edgeMap;
  {
    size_t blockStart = xmlText.find("<EDGE");
    size_t blockEnd = xmlText.find("</EDGE>");
    if (blockStart != std::string::npos && blockEnd != std::string::npos && blockEnd > blockStart)
    {
      const std::string block = xmlText.substr(blockStart, blockEnd - blockStart);
      std::regex edge_re(R"(<E\s+ID=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</E>)", std::regex::icase);
      auto begin = std::sregex_iterator(block.begin(), block.end(), edge_re);
      auto end = std::sregex_iterator();
      for (auto it = begin; it != end; ++it)
      {
        const int id = std::atoi((*it)[1].str().c_str());
        std::istringstream iss((*it)[2].str());
        EdgeInfo e;
        if (iss >> e.v0 >> e.v1) edgeMap[id] = e;
      }
    }
  }
  auto pair_key = [](int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<long long>(a) << 32) | static_cast<unsigned int>(b);
  };
  std::unordered_map<long long, int> edgeLookup;
  for (const auto& kv : edgeMap)
  {
    edgeLookup[pair_key(kv.second.v0, kv.second.v1)] = kv.first;
  }

  struct FaceInfo
  {
    int id{-1};
    char type{'Q'}; // 'Q' quad, 'T' tri
    std::vector<int> edges;
    std::vector<int> verts;
  };
  std::unordered_map<int, FaceInfo> faceMap;
  {
    size_t faceStart = xmlText.find("<FACE");
    size_t faceEnd = xmlText.find("</FACE>");
    if (faceStart != std::string::npos && faceEnd != std::string::npos && faceEnd > faceStart)
    {
      const std::string block = xmlText.substr(faceStart, faceEnd - faceStart);
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
          // reconstruct vertex loop from edges
          auto eIt = edgeMap.find(f.edges.front());
          if (eIt != edgeMap.end())
          {
            f.verts.push_back(eIt->second.v0);
            f.verts.push_back(eIt->second.v1);
            for (size_t idx = 1; idx < f.edges.size(); ++idx)
            {
              auto e2 = edgeMap.find(f.edges[idx]);
              if (e2 == edgeMap.end()) continue;
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
          faceMap[f.id] = f;
        }
      }
    }
  }

  struct ElemHex
  {
    int id{0};
    std::array<int, 8> verts{};
    std::array<int, 12> edges{};
    std::array<int, 6> faces{};
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
  struct ElemQuad { int id{0}; std::array<int, 4> verts{}; };
  struct ElemTri { int id{0}; std::array<int, 3> verts{}; };

  auto init_array = [](auto& arr) { for (auto& v : arr) v = -1; };

  struct InitExtras
  {
    void operator()(ElemPrism& elem) const
    {
      for (size_t i = 0; i < elem.edges.size(); ++i) elem.edges[i] = -1;
      for (size_t i = 0; i < elem.faces.size(); ++i) elem.faces[i] = -1;
      elem.valid = false;
    }
    void operator()(ElemHex&) const {}
    void operator()(ElemQuad&) const {}
    void operator()(ElemTri&) const {}
  } init_extras;

  std::vector<ElemHex> hex_conn;
  std::vector<ElemPrism> prism_conn;
  std::vector<ElemQuad> quad_conn;
  std::vector<ElemTri> tri_conn;

  const int hexEdgeVerts[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {0, 4}, {1, 5}, {2, 6}, {3, 7},
      {4, 5}, {5, 6}, {6, 7}, {7, 4}};

  auto build_hex_from_faces = [&](const std::array<int, 6>& faceIds, ElemHex& out) -> bool
  {
    out.faces = faceIds;
    auto fBottom = faceMap.find(faceIds[0]);
    auto fTop = faceMap.find(faceIds[5]);
    if (fBottom == faceMap.end() || fTop == faceMap.end() ||
        fBottom->second.verts.size() < 4 || fTop->second.verts.size() < 4)
      return false;
    std::array<int, 4> bottom{};
    std::array<int, 4> top{};
    for (int i = 0; i < 4; ++i)
    {
      bottom[i] = fBottom->second.verts[i];
      top[i] = fTop->second.verts[i];
      out.verts[i] = bottom[i];
    }
    std::unordered_map<int, int> topSet;
    for (int v : top) topSet[v] = 1;

    std::vector<int> elemEdges;
    for (int fid : faceIds)
    {
      auto ft = faceMap.find(fid);
      if (ft != faceMap.end()) elemEdges.insert(elemEdges.end(), ft->second.edges.begin(), ft->second.edges.end());
    }

    std::array<int, 4> topForBottom{};
    topForBottom.fill(-1);
    for (int b = 0; b < 4; ++b)
    {
      const int bVert = bottom[b];
      for (int eid : elemEdges)
      {
        auto eIt = edgeMap.find(eid);
        if (eIt == edgeMap.end()) continue;
        const int a = eIt->second.v0;
        const int c = eIt->second.v1;
        if (a == bVert && topSet.count(c))
        {
          topForBottom[b] = c;
          break;
        }
        if (c == bVert && topSet.count(a))
        {
          topForBottom[b] = a;
          break;
        }
      }
      if (topForBottom[b] == -1) topForBottom[b] = top[b];
    }
    for (int i = 0; i < 4; ++i) out.verts[4 + i] = topForBottom[i];

    for (int e = 0; e < 12; ++e)
    {
      const int a = out.verts[hexEdgeVerts[e][0]];
      const int b = out.verts[hexEdgeVerts[e][1]];
      auto it = edgeLookup.find(pair_key(a, b));
      out.edges[e] = (it != edgeLookup.end()) ? it->second : -1;
    }
    out.valid = true;
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
      hex.id = static_cast<int>(hex_conn.size());
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
          auto it = edgeLookup.find(pair_key(hex.verts[hexEdgeVerts[e][0]], hex.verts[hexEdgeVerts[e][1]]));
          hex.edges[e] = (it != edgeLookup.end()) ? it->second : -1;
        }
        hex.valid = true;
      }
      if (hex.valid) hex_conn.push_back(hex);
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
      init_extras(elem);
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

  parse_simple("P", 6, prism_conn);
  parse_simple("PRISM", 6, prism_conn);
  parse_simple("Q", 4, quad_conn);
  parse_simple("QUAD", 4, quad_conn);
  parse_simple("T", 3, tri_conn);
  parse_simple("TRI", 3, tri_conn);

  auto find_face_id = [&](const std::vector<int>& vertsWanted, char type)->int
  {
    for (const auto& kv : faceMap)
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

  // 填充棱柱的边/面索引，便于曲边/曲面处理
  for (auto& pri : prism_conn)
  {
    if (pri.verts[0] < 0) continue;
    init_array(pri.edges);
    const int edgePairs[9][2] = {{0,1},{1,2},{2,0},{3,4},{4,5},{5,3},{0,3},{1,4},{2,5}};
    for (int e = 0; e < 9; ++e)
    {
      int a = pri.verts[edgePairs[e][0]];
      int b = pri.verts[edgePairs[e][1]];
      auto itE = edgeLookup.find(pair_key(a, b));
      if (itE != edgeLookup.end()) pri.edges[e] = itE->second;
    }
    init_array(pri.faces);
    pri.faces[0] = find_face_id({pri.verts[0], pri.verts[1], pri.verts[2]}, 'T'); // 底三角
    pri.faces[1] = find_face_id({pri.verts[3], pri.verts[4], pri.verts[5]}, 'T'); // 顶三角
    pri.faces[2] = find_face_id({pri.verts[0], pri.verts[1], pri.verts[4], pri.verts[3]}, 'Q');
    pri.faces[3] = find_face_id({pri.verts[1], pri.verts[2], pri.verts[5], pri.verts[4]}, 'Q');
    pri.faces[4] = find_face_id({pri.verts[2], pri.verts[0], pri.verts[3], pri.verts[5]}, 'Q');
    pri.valid = true;
  }

  if (hex_conn.empty() && prism_conn.empty() && quad_conn.empty() && tri_conn.empty())
  {
    std::cerr << "XML/FLD loader: no supported elements found." << std::endl;
    return false;
  }

  // --- CURVED data ---
  struct CurvedEdge { unsigned int numPts{0}; std::vector<float3> pts; };
  struct CurvedFace { unsigned int numPts{0}; char type{'Q'}; std::vector<float3> pts; };
  std::unordered_map<int, CurvedEdge> curvedEdges;
  std::unordered_map<int, CurvedFace> curvedFaces;
  {
    size_t cStart = xmlText.find("<CURVED");
    size_t cEnd = xmlText.find("</CURVED>");
    if (cStart != std::string::npos && cEnd != std::string::npos && cEnd > cStart)
    {
      const std::string block = xmlText.substr(cStart, cEnd - cStart);
      std::regex edge_re(R"(<E[^>]*EDGEID=\"?(\d+)\"?[^>]*NUMPOINTS=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</E>)", std::regex::icase);
      auto be = std::sregex_iterator(block.begin(), block.end(), edge_re);
      auto en = std::sregex_iterator();
      for (auto it = be; it != en; ++it)
      {
        CurvedEdge ce;
        ce.numPts = static_cast<unsigned int>(std::atoi((*it)[2].str().c_str()));
        const int eid = std::atoi((*it)[1].str().c_str());
        std::istringstream iss((*it)[3].str());
        float3 p{};
        while (iss >> p.x >> p.y >> p.z) ce.pts.push_back(p);
        if (!ce.pts.empty()) curvedEdges[eid] = ce;
      }

      std::regex face_re(R"(<F[^>]*FACEID=\"?(\d+)\"?[^>]*NUMPOINTS=\"?(\d+)\"?[^>]*>\s*([^<]+)\s*</F>)", std::regex::icase);
      auto bf = std::sregex_iterator(block.begin(), block.end(), face_re);
      for (auto it = bf; it != en; ++it)
      {
        CurvedFace cf;
        cf.numPts = static_cast<unsigned int>(std::atoi((*it)[2].str().c_str()));
        const int fid = std::atoi((*it)[1].str().c_str());
        std::istringstream iss((*it)[3].str());
        float3 p{};
        while (iss >> p.x >> p.y >> p.z) cf.pts.push_back(p);
        if (!cf.pts.empty()) curvedFaces[fid] = cf;
      }
    }
  }

  // --- COMPOSITE 和 EXPANSIONS 解析（简化版，只做元素顺序映射） ---
  std::unordered_map<int, std::vector<std::pair<char,int>>> composites; // Cid -> [(type,id)]
  {
    std::regex comp_re(R"(<C\s+ID=\"?(\d+)\"?\s*>\s*([^<]+)\s*</C>)", std::regex::icase);
    auto begin = std::sregex_iterator(xmlText.begin(), xmlText.end(), comp_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      int cid = std::atoi((*it)[1].str().c_str());
      std::string body = (*it)[2].str();
      std::regex entry_re(R"(([HQPT])\s*\[\s*([0-9\-\:, ]+)\s*\])", std::regex::icase);
      auto b2 = std::sregex_iterator(body.begin(), body.end(), entry_re);
      for (auto jt = b2; jt != end; ++jt) {
        char typ = static_cast<char>(std::toupper((*jt)[1].str()[0]));
        std::string range = (*jt)[2].str();
        // parse list/range e.g. 0-5 or 1,2,3
        std::stringstream ss(range);
        std::string token;
        while (std::getline(ss, token, ',')) {
          int lo=0, hi=0;
          if (token.find('-') != std::string::npos) {
            sscanf(token.c_str(), "%d-%d", &lo, &hi);
          } else {
            lo = hi = std::atoi(token.c_str());
          }
          for (int v = lo; v <= hi; ++v) {
            composites[cid].push_back({typ, v});
          }
        }
      }
    }
  }

  struct ExpansionSpec
  {
    std::vector<std::pair<char,int>> elems;
    uint3 modes{make_uint3(0,0,0)};
    std::vector<std::string> fields;
  };
  std::vector<ExpansionSpec> expansion_specs;
  {
    std::regex exp_re(R"(<E\s+([^>]*)>)", std::regex::icase);
    auto begin = std::sregex_iterator(xmlText.begin(), xmlText.end(), exp_re);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it)
    {
      const std::string header = (*it)[1].str();
      const std::string compositeAttr = GetAttribute("<E " + header + ">", "COMPOSITE");
      const std::string numModesPerDir = GetAttribute("<E " + header + ">", "NUMMODESPERDIR");
      const std::string numModesStr = GetAttribute("<E " + header + ">", "NUMMODES");
      const std::string fieldsAttr = GetAttribute("<E " + header + ">", "FIELDS");
      ExpansionSpec spec;
      spec.modes = !numModesPerDir.empty() ? ParseModes(numModesPerDir)
                                           : (!numModesStr.empty() ? ParseModes(numModesStr) : modes);
      if (!fieldsAttr.empty()) spec.fields = ParseFieldList(fieldsAttr);

      std::regex cid_re(R"(C\[\s*([0-9\-]+)\s*\])", std::regex::icase);
      auto bc = std::sregex_iterator(compositeAttr.begin(), compositeAttr.end(), cid_re);
      for (auto ct = bc; ct != end; ++ct) {
        std::string token = (*ct)[1].str();
        int lo=0, hi=0;
        if (token.find('-') != std::string::npos) {
          sscanf(token.c_str(), "%d-%d", &lo, &hi);
        } else {
          lo = hi = std::atoi(token.c_str());
        }
        for (int cid = lo; cid <= hi; ++cid) {
          auto itc = composites.find(cid);
          if (itc != composites.end()) {
            spec.elems.insert(spec.elems.end(), itc->second.begin(), itc->second.end());
          }
        }
      }
      if (!spec.elems.empty()) expansion_specs.push_back(spec);
    }
  }

  // --- coefficients & FIELDS 列表 ---
  std::vector<float> coeffs;
  std::vector<uint8_t> coeff_bytes;
  std::vector<std::string> fieldNames;
  const std::string fldText = ReadFile(fld_path);
  if (!fldText.empty())
  {
    std::regex fields_re(R"(FIELDS=\"([^\"]+)\")", std::regex::icase);
    std::smatch m;
    if (std::regex_search(fldText, m, fields_re))
    {
      std::string body = m[1].str();
      std::stringstream ss(body);
      std::string tok;
      while (std::getline(ss, tok, ','))
      {
        std::string name = Trim(tok);
        if (!name.empty()) fieldNames.push_back(name);
      }
    }
    const std::string payload = LargestPayloadBetweenTags(fldText);
    if (!payload.empty())
    {
      ParseAsciiFloats(payload, coeffs);
      if (coeffs.empty())
      {
        const std::vector<uint8_t> decoded = Base64Decode(payload);
        std::vector<uint8_t> inflated;
        if (!decoded.empty())
        {
          const bool looksCompressed = decoded.size() > 2 && decoded[0] == 0x78;
          if (!DecompressZlib(decoded, inflated))
          {
            if (looksCompressed)
            {
              std::cerr << "XML/FLD loader: coefficient payload appears zlib-compressed but zlib support is unavailable; modal field data will be skipped." << std::endl;
            }
            else
            {
              inflated = decoded; // maybe stored uncompressed base64
            }
          }
          if (!inflated.empty()) coeff_bytes = std::move(inflated);
        }
      }
    }
  }
  if (fieldNames.empty()) fieldNames.push_back("u");

  // --- build scene ---
  const uint2 quadModes2 = make_uint2(modes.x, modes.y);
  const uint2 triModes2 = make_uint2(modes.x, modes.y);

  // 构建 ID -> 索引 映射，便于从 expansion 顺序找到数组下标
  std::unordered_map<int,size_t> hex_id_to_idx, prism_id_to_idx, quad_id_to_idx, tri_id_to_idx;
  for (size_t i=0;i<hex_conn.size();++i) hex_id_to_idx[hex_conn[i].id] = i;
  for (size_t i=0;i<prism_conn.size();++i) prism_id_to_idx[prism_conn[i].id] = i;
  for (size_t i=0;i<quad_conn.size();++i) quad_id_to_idx[quad_conn[i].id] = i;
  for (size_t i=0;i<tri_conn.size();++i) tri_id_to_idx[tri_conn[i].id] = i;

  auto fetch_vertex = [&](int vid) -> float3 {
    if (vid >= 0 && vid < static_cast<int>(vertices.size())) return vertices[vid];
    return make_float3(0, 0, 0);
  };

  auto build_hex_geometry = [&](const ElemHex& elem,
                                const uint3& baseModes,
                                std::vector<float>& gx,
                                std::vector<float>& gy,
                                std::vector<float>& gz,
                                uint3& geomModes) -> bool
  {
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

    float3 corners[8];
    for (int i = 0; i < 8; ++i) corners[i] = fetch_vertex(elem.verts[i]);
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

    // edge overrides
    const bool reversePlacement[12] = {false, false, true, true, false, false, false, false, false, false, true, true};
    for (int e = 0; e < 12; ++e)
    {
      const int globalE = elem.edges[e];
      if (globalE < 0) continue;
      auto ce = curvedEdges.find(globalE);
      if (ce == curvedEdges.end() || ce->second.pts.empty()) continue;
      auto eInfo = edgeMap.find(globalE);
      bool reverse = reversePlacement[e];
      if (eInfo != edgeMap.end())
      {
        const int localStart = elem.verts[hexEdgeVerts[e][0]];
        const int localEnd = elem.verts[hexEdgeVerts[e][1]];
        if (eInfo->second.v0 == localEnd && eInfo->second.v1 == localStart) reverse = !reverse;
      }
      unsigned int targetCount = (e == 1 || e == 3 || e == 9 || e == 11) ? geomModes.y
                              : (e == 4 || e == 5 || e == 6 || e == 7) ? geomModes.z
                              : geomModes.x;
      const std::vector<float3> edgePts = ResampleEdgePoints(ce->second.pts, targetCount, reverse);
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

    auto face_corners = [&](int f) -> std::array<float3, 4>
    {
      switch (f)
      {
        case 0: return {fetch_vertex(elem.verts[0]), fetch_vertex(elem.verts[1]), fetch_vertex(elem.verts[2]), fetch_vertex(elem.verts[3])};
        case 1: return {fetch_vertex(elem.verts[0]), fetch_vertex(elem.verts[1]), fetch_vertex(elem.verts[5]), fetch_vertex(elem.verts[4])};
        case 2: return {fetch_vertex(elem.verts[1]), fetch_vertex(elem.verts[2]), fetch_vertex(elem.verts[6]), fetch_vertex(elem.verts[5])};
        case 3: return {fetch_vertex(elem.verts[3]), fetch_vertex(elem.verts[2]), fetch_vertex(elem.verts[6]), fetch_vertex(elem.verts[7])};
        case 4: return {fetch_vertex(elem.verts[0]), fetch_vertex(elem.verts[3]), fetch_vertex(elem.verts[7]), fetch_vertex(elem.verts[4])};
        case 5: return {fetch_vertex(elem.verts[4]), fetch_vertex(elem.verts[5]), fetch_vertex(elem.verts[6]), fetch_vertex(elem.verts[7])};
        default: return {make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0),make_float3(0,0,0)};
      }
    };

    // face overrides (quad faces only)
    for (int lf = 0; lf < 6; ++lf)
    {
      const int globalF = elem.faces[lf];
      auto cf = curvedFaces.find(globalF);
      if (cf == curvedFaces.end() || cf->second.pts.empty()) continue;
      unsigned int dimU = (lf == 1 || lf == 3) ? geomModes.x : (lf == 2 || lf == 4) ? geomModes.y : geomModes.x;
      unsigned int dimV = (lf == 1 || lf == 3) ? geomModes.z : (lf == 2 || lf == 4) ? geomModes.z : geomModes.y;
      const std::array<float3, 4> cornersArr = face_corners(lf);
      std::vector<float3> faceGrid = ResampleQuadFace(cf->second.pts, dimU, dimV, cornersArr);
      if (faceGrid.size() != static_cast<size_t>(dimU) * dimV) continue;
      if (lf == 0)
      {
        for (unsigned int j = 0; j < dimV && j < static_cast<unsigned int>(ny); ++j)
          for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
            nodal[idx(i, j, 0)] = faceGrid[j * dimU + i];
      }
      else if (lf == 5)
      {
        for (unsigned int j = 0; j < dimV && j < static_cast<unsigned int>(ny); ++j)
          for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
            nodal[idx(i, j, nz - 1)] = faceGrid[j * dimU + i];
      }
      else if (lf == 1)
      {
        for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
          for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
            nodal[idx(i, 0, k)] = faceGrid[k * dimU + i];
      }
      else if (lf == 3)
      {
        for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
          for (unsigned int i = 0; i < dimU && i < static_cast<unsigned int>(nx); ++i)
            nodal[idx(i, ny - 1, k)] = faceGrid[k * dimU + i];
      }
      else if (lf == 2)
      {
        for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
          for (unsigned int j = 0; j < dimU && j < static_cast<unsigned int>(ny); ++j)
            nodal[idx(nx - 1, j, k)] = faceGrid[k * dimU + j];
      }
      else if (lf == 4)
      {
        for (unsigned int k = 0; k < dimV && k < static_cast<unsigned int>(nz); ++k)
          for (unsigned int j = 0; j < dimU && j < static_cast<unsigned int>(ny); ++j)
            nodal[idx(0, j, k)] = faceGrid[k * dimU + j];
      }
    }

    if (!TensorModalFit3D(nodal, rNodes, sNodes, tNodes, gx, gy, gz)) return false;
    return true;
  };

  auto build_quad_geometry = [&](const ElemQuad& elem,
                                 const uint2& baseModes,
                                 std::vector<float>& gx,
                                 std::vector<float>& gy,
                                 std::vector<float>& gz,
                                 uint2& geomModes) -> bool
  {
    unsigned int maxOrder = std::max(baseModes.x, baseModes.y);
    for (int e = 0; e < 4; ++e)
    {
      int va = elem.verts[e];
      int vb = elem.verts[(e + 1) % 4];
      auto itEdge = edgeLookup.find(pair_key(va, vb));
      if (itEdge != edgeLookup.end())
      {
        auto ce = curvedEdges.find(itEdge->second);
        if (ce != curvedEdges.end()) maxOrder = std::max(maxOrder, ce->second.numPts);
      }
    }
    // 若存在曲面采样点，使用其阶数更新目标阶
    auto cfIt = curvedFaces.find(elem.id);
    if (cfIt != curvedFaces.end() && !cfIt->second.pts.empty())
    {
      const unsigned int n = static_cast<unsigned int>(std::round(
        std::sqrt(static_cast<double>(cfIt->second.pts.size()))));
      if (n > 1) maxOrder = std::max(maxOrder, n);
    }
    geomModes = make_uint2(std::max(baseModes.x, maxOrder), std::max(baseModes.y, maxOrder));
    const int nx = static_cast<int>(geomModes.x);
    const int ny = static_cast<int>(geomModes.y);
    const std::vector<float> rNodes = GLLNodes(geomModes.x);
    const std::vector<float> sNodes = GLLNodes(geomModes.y);
    auto idx = [&](int i, int j) { return j * nx + i; };

    float3 c0 = fetch_vertex(elem.verts[0]);
    float3 c1 = fetch_vertex(elem.verts[1]);
    float3 c2 = fetch_vertex(elem.verts[2]);
    float3 c3 = fetch_vertex(elem.verts[3]);

    // 若有完整曲面网格，直接重采样并拟合
    if (cfIt != curvedFaces.end() && !cfIt->second.pts.empty())
    {
      std::array<float3, 4> corners = {c0, c1, c2, c3};
      std::vector<float3> faceGrid = ResampleQuadFace(cfIt->second.pts,
                                                      static_cast<unsigned int>(nx),
                                                      static_cast<unsigned int>(ny),
                                                      corners);
      if (faceGrid.size() == static_cast<size_t>(nx * ny))
      {
        return TensorModalFit2D(faceGrid, rNodes, sNodes, gx, gy, gz);
      }
    }

    std::array<std::vector<float3>, 4> edges;
    edges[0].resize(nx);
    edges[2].resize(nx);
    edges[1].resize(ny);
    edges[3].resize(ny);
    for (int i = 0; i < nx; ++i)
    {
      const float u = (nx == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(nx - 1);
      edges[0][i] = make_float3(c0.x + u * (c1.x - c0.x),
                                c0.y + u * (c1.y - c0.y),
                                c0.z + u * (c1.z - c0.z));
      edges[2][i] = make_float3(c3.x + u * (c2.x - c3.x),
                                c3.y + u * (c2.y - c3.y),
                                c3.z + u * (c2.z - c3.z));
    }
    for (int j = 0; j < ny; ++j)
    {
      const float v = (ny == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(ny - 1);
      edges[3][j] = make_float3(c0.x + v * (c3.x - c0.x),
                                c0.y + v * (c3.y - c0.y),
                                c0.z + v * (c3.z - c0.z));
      edges[1][j] = make_float3(c1.x + v * (c2.x - c1.x),
                                c1.y + v * (c2.y - c1.y),
                                c1.z + v * (c2.z - c1.z));
    }

    for (int e = 0; e < 4; ++e)
    {
      int va = elem.verts[e];
      int vb = elem.verts[(e + 1) % 4];
      auto itEdge = edgeLookup.find(pair_key(va, vb));
      if (itEdge == edgeLookup.end()) continue;
      auto ce = curvedEdges.find(itEdge->second);
      if (ce == curvedEdges.end() || ce->second.pts.empty()) continue;
      const bool reverse = (vb < va); // heuristic
      std::vector<float3> pts = ResampleEdgePoints(ce->second.pts, (e % 2 == 0) ? geomModes.x : geomModes.y, reverse);
      if (pts.empty()) continue;
      if (e == 0) edges[0] = pts;
      else if (e == 2) edges[2] = pts;
      else if (e == 1) edges[1] = pts;
      else edges[3] = pts;
    }

    std::vector<float3> nodal(static_cast<size_t>(nx) * ny);
    for (int j = 0; j < ny; ++j)
    {
      const float v = (ny == 1) ? 0.0f : static_cast<float>(j) / static_cast<float>(ny - 1);
      for (int i = 0; i < nx; ++i)
      {
        const float u = (nx == 1) ? 0.0f : static_cast<float>(i) / static_cast<float>(nx - 1);
        const float3 Cb = edges[0][i];
        const float3 Ct = edges[2][i];
        const float3 Cl = edges[3][j];
        const float3 Cr = edges[1][j];
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
  };

  auto build_tri_geometry = [&](const ElemTri& elem,
                                const uint2& baseModes,
                                std::vector<float>& gx,
                                std::vector<float>& gy,
                                std::vector<float>& gz,
                                uint2& geomModes) -> bool
  {
    // 几何阶取场阶与曲边阶的最大值
    unsigned int maxOrder = std::max(baseModes.x, baseModes.y);
    const int triEdge[3][2] = {{0,1},{1,2},{2,0}};
    for (int e = 0; e < 3; ++e)
    {
      int a = elem.verts[triEdge[e][0]];
      int b = elem.verts[triEdge[e][1]];
      auto itE = edgeLookup.find(pair_key(a, b));
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

    const float3 v0 = fetch_vertex(elem.verts[0]);
    const float3 v1 = fetch_vertex(elem.verts[1]);
    const float3 v2 = fetch_vertex(elem.verts[2]);
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
      if (localEdge == 0) // 0-1: j=0
      {
        for (auto& nr : nodeRefs) if (nr.j == 0) idxList.push_back(nr.idx);
      }
      else if (localEdge == 1) // 1-2: i+j=order-1
      {
        for (auto& nr : nodeRefs) if (nr.i + nr.j == order - 1) idxList.push_back(nr.idx);
      }
      else if (localEdge == 2) // 2-0: i=0
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
      auto itE = edgeLookup.find(pair_key(a, b));
      if (itE == edgeLookup.end()) continue;
      auto ce = curvedEdges.find(itE->second);
      if (ce == curvedEdges.end()) continue;
      bool reverse = false;
      auto eInfo = edgeMap.find(itE->second);
      if (eInfo != edgeMap.end())
      {
        if (eInfo->second.v0 == b && eInfo->second.v1 == a) reverse = true;
      }
      std::vector<float3> samples = ResampleEdgePoints(ce->second.pts, order, reverse);
      set_edge(e, samples);
    }

    return SolveTriModalCoeffs(nodal, nodesRS, geomModes, gx, gy, gz);
  };

  auto build_prism_geometry = [&](const ElemPrism& elem,
                                  const uint3& baseModes,
                                  std::vector<float>& gx,
                                  std::vector<float>& gy,
                                  std::vector<float>& gz,
                                  uint3& geomModes) -> bool
  {
    // 基于边阶数取几何阶；若无曲边则退化为线性几何
    unsigned int maxOrder = std::max({baseModes.x, baseModes.y, baseModes.z});
    const int edgePairs[9][2] = {{0,1},{1,2},{2,0},{3,4},{4,5},{5,3},{0,3},{1,4},{2,5}};
    for (int e = 0; e < 9; ++e)
    {
      int globalE = elem.edges[e];
      if (globalE < 0)
      {
        int a = elem.verts[edgePairs[e][0]];
        int b = elem.verts[edgePairs[e][1]];
        auto itE = edgeLookup.find(pair_key(a, b));
        if (itE != edgeLookup.end()) globalE = itE->second;
      }
      auto itC = curvedEdges.find(globalE);
      if (itC != curvedEdges.end()) maxOrder = std::max(maxOrder, itC->second.numPts);
    }
    // 曲面阶
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
                           std::max(maxOrder, std::max(1u, baseModes.x))); // 保证 z 阶不小于 r 阶

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
    for (int i = 0; i < 6; ++i) v[i] = fetch_vertex(elem.verts[i]);

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

    auto apply_edge = [&](int localEdge, const std::vector<float3>& samples)
    {
      if (samples.empty()) return;
      std::vector<size_t> idxList;
      switch (localEdge)
      {
        case 0: // 0-1, s=-1,t=-1, r varies
          for (unsigned int ir = 0; ir < nx; ++ir)
            if (keyToIdx.count(PrismNodeKey(ir, 0, 0)))
              idxList.push_back(keyToIdx[PrismNodeKey(ir, 0, 0)]);
          break;
        case 1: // 1-2, r=max, t=-1, s varies
          for (unsigned int js = 0; js < ny; ++js)
            if (keyToIdx.count(PrismNodeKey(nx - 1, js, 0)))
              idxList.push_back(keyToIdx[PrismNodeKey(nx - 1, js, 0)]);
          break;
        case 6: // 0-3, r=-1,s=-1, t varies
          for (unsigned int kt = 0; kt < nz; ++kt)
            if (keyToIdx.count(PrismNodeKey(0, 0, kt)))
              idxList.push_back(keyToIdx[PrismNodeKey(0, 0, kt)]);
          break;
        case 7: // 1-4, r=max,s=-1, t varies
          for (unsigned int kt = 0; kt < nz - (nx - 1); ++kt)
            if (keyToIdx.count(PrismNodeKey(nx - 1, 0, kt)))
              idxList.push_back(keyToIdx[PrismNodeKey(nx - 1, 0, kt)]);
          break;
        case 8: // 2-5, r=max, s=max, t varies
          for (unsigned int kt = 0; kt < nz - (nx - 1); ++kt)
            if (keyToIdx.count(PrismNodeKey(nx - 1, ny - 1, kt)))
              idxList.push_back(keyToIdx[PrismNodeKey(nx - 1, ny - 1, kt)]);
          break;
        default:
          return;
      }
      if (idxList.empty()) return;
      std::vector<float3> resampled = ResampleEdgePoints(samples, static_cast<unsigned int>(idxList.size()));
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
        auto itE = edgeLookup.find(pair_key(a, b));
        if (itE != edgeLookup.end()) globalE = itE->second;
      }
      if (globalE < 0) continue;
      auto itC = curvedEdges.find(globalE);
      if (itC == curvedEdges.end()) continue;
      apply_edge(e, itC->second.pts);
    }

    // 三角面曲面：底(0-1-2)顶(3-4-5)
    auto apply_tri_face = [&](int faceIdx, bool top)
    {
      int fid = elem.faces[faceIdx];
      if (fid < 0) return;
      auto cf = curvedFaces.find(fid);
      if (cf == curvedFaces.end() || cf->second.pts.empty()) return;
      unsigned int order = std::min({geomModes.x, geomModes.y});
      if (order < 2) return;
      std::array<float3,3> corners = top ?
        std::array<float3,3>{fetch_vertex(elem.verts[3]), fetch_vertex(elem.verts[4]), fetch_vertex(elem.verts[5])} :
        std::array<float3,3>{fetch_vertex(elem.verts[0]), fetch_vertex(elem.verts[1]), fetch_vertex(elem.verts[2])};
      std::vector<float3> triGrid = ResampleTriFace(cf->second.pts, order, corners);
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

    // 四边形侧面曲面：face2 0-1-4-3 (s=-1)，face3 1-2-5-4 (r=+1)，face4 2-0-3-5 (r=-1)
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
      std::vector<float3> faceGrid = ResampleQuadFace(cf->second.pts, dimU, dimV, corners);
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
                    {fetch_vertex(elem.verts[0]), fetch_vertex(elem.verts[1]),
                     fetch_vertex(elem.verts[4]), fetch_vertex(elem.verts[3])});
    apply_quad_face(3, ny, (nz > 0) ? nz - 1 : nz, (nx > 0) ? nx - 1 : 0, -1,
                    {fetch_vertex(elem.verts[1]), fetch_vertex(elem.verts[2]),
                     fetch_vertex(elem.verts[5]), fetch_vertex(elem.verts[4])});
    apply_quad_face(4, ny, nz, 0, -1,
                    {fetch_vertex(elem.verts[2]), fetch_vertex(elem.verts[0]),
                     fetch_vertex(elem.verts[3]), fetch_vertex(elem.verts[5])});

    // 构造 Vandermonde 并反求模态系数
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
  };

  // 构建 expansion 顺序记录（含变量阶与字段列表）
  struct ElemRecord
  {
    char typ{'H'};
    int id{-1};
    uint3 modes{make_uint3(0,0,0)};
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
  if (!expansion_specs.empty())
  {
    for (const auto& spec : expansion_specs) append_records(spec);
  }
  else
  {
    ExpansionSpec fallback;
    fallback.modes = modes;
    fallback.fields = fieldNames;
    for (const auto& h : hex_conn) fallback.elems.push_back({'H', h.id});
    for (const auto& p : prism_conn) fallback.elems.push_back({'P', p.id});
    for (const auto& q : quad_conn) fallback.elems.push_back({'Q', q.id});
    for (const auto& t : tri_conn) fallback.elems.push_back({'T', t.id});
    append_records(fallback);
  }

  // 若存在 <EXPANSIONS>，仅保留其中引用到的元素，避免把 <FACE> 中的 Q/T 当成独立单元
  if (!expansion_specs.empty())
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
    filter_by_set(hex_conn, used_hex);
    filter_by_set(prism_conn, used_prism);
    filter_by_set(quad_conn, used_quad);
    filter_by_set(tri_conn, used_tri);
  }

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

  const auto expected_coeff_total = [&]() -> size_t
  {
    size_t total = 0;
    for (const auto& fname : fieldNames)
    {
      for (const auto& rec : elem_records)
      {
        const std::vector<std::string>& fList = rec.fields.empty() ? fieldNames : rec.fields;
        if (std::find(fList.begin(), fList.end(), fname) == fList.end()) continue;
        total += coeff_count(rec);
      }
    }
    return total;
  };

  if (coeffs.empty() && !coeff_bytes.empty())
  {
    const size_t expected = expected_coeff_total();
    const size_t doubleCount = coeff_bytes.size() / sizeof(double);
    const size_t floatCount = coeff_bytes.size() / sizeof(float);
    bool useDouble = false;
    if (expected > 0)
    {
      if (doubleCount == expected) useDouble = true;
      else if (floatCount == expected) useDouble = false;
      else if (doubleCount > 0 && floatCount == expected * 2) useDouble = true; // common double->float confusion
    }
    else if (doubleCount > 0 && floatCount == doubleCount * 2)
    {
      useDouble = true;
    }

    if (useDouble && coeff_bytes.size() % sizeof(double) == 0)
    {
      coeffs.resize(doubleCount);
      const double* src = reinterpret_cast<const double*>(coeff_bytes.data());
      for (size_t i = 0; i < doubleCount; ++i) coeffs[i] = static_cast<float>(src[i]);
    }
    else if (coeff_bytes.size() % sizeof(float) == 0)
    {
      coeffs.resize(floatCount);
      std::memcpy(coeffs.data(), coeff_bytes.data(), coeff_bytes.size());
    }
  }

  // 为每个元素记录多场切片
  std::vector<std::vector<Scene::FieldSlice>> hex_slices(hex_conn.size());
  std::vector<std::vector<Scene::FieldSlice>> prism_slices(prism_conn.size());
  std::vector<std::vector<Scene::FieldSlice>> quad_slices(quad_conn.size());
  std::vector<std::vector<Scene::FieldSlice>> tri_slices(tri_conn.size());

  // 为几何基阶准备映射，取同一单元所有记录中的最大阶
  auto max3 = [](uint3 a, uint3 b)->uint3 {
    return make_uint3(std::max(a.x,b.x), std::max(a.y,b.y), std::max(a.z,b.z));
  };
  auto max2 = [](uint2 a, uint2 b)->uint2 {
    return make_uint2(std::max(a.x,b.x), std::max(a.y,b.y));
  };
  std::unordered_map<int,uint3> hex_mode_map;
  std::unordered_map<int,uint3> prism_mode_map;
  std::unordered_map<int,uint2> quad_mode_map;
  std::unordered_map<int,uint2> tri_mode_map;
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
  for (const auto& fname : fieldNames)
  {
    size_t cursor = coeff_base;
    bool coeff_stream_exhausted = false;
    size_t needed_at_break = 0;
    for (const auto& rec : elem_records)
    {
      const std::vector<std::string>& fList = rec.fields.empty() ? fieldNames : rec.fields;
      if (std::find(fList.begin(), fList.end(), fname) == fList.end()) continue;
      const size_t cnt = coeff_count(rec);
      if (cnt == 0) continue;
      needed_at_break = cursor + cnt;
      coeffs_required = std::max(coeffs_required, needed_at_break);
      if (cursor + cnt > coeffs.size())
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
                << "' (have " << coeffs.size() << ", need at least " << needed_at_break
                << "); remaining elements will fall back to defaults." << std::endl;
    }
    coeff_base = cursor;
  }

  auto first_slice_ptr = [&](const std::vector<Scene::FieldSlice>& vec)->const Scene::FieldSlice*
  {
    if (vec.empty()) return nullptr;
    return &vec.front();
  };

  for (size_t i = 0; i < hex_conn.size(); ++i) {
    const Scene::FieldSlice* fs = first_slice_ptr(hex_slices[i]);
    const float* cptr = (fs && fs->offset >=0 && static_cast<size_t>(fs->offset + fs->count) <= coeffs.size())
                        ? coeffs.data() + fs->offset : nullptr;
    uint3 fm = modes;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes3; fcnt = fs->count; }
    zidingyi::HexElementData h{};
    for (int v = 0; v < 8; ++v) h.vertices[v] = fetch_vertex(hex_conn[i].verts[v]);
    h.fieldModes = fm;
    std::vector<float> gx, gy, gz;
    uint3 geomModes = make_uint3(0,0,0);
    uint3 baseGeom = modes;
    auto itBase = hex_mode_map.find(hex_conn[i].id);
    if (itBase != hex_mode_map.end()) baseGeom = itBase->second;
    build_hex_geometry(hex_conn[i], baseGeom, gx, gy, gz, geomModes);
    scene.add_hex(h, cptr, fm, fcnt,
                  gx.empty() ? nullptr : gx.data(),
                  gy.empty() ? nullptr : gy.data(),
                  gz.empty() ? nullptr : gz.data(),
                  geomModes,
                  gx.size(),
                  &hex_slices[i]);
  }
  for (size_t i = 0; i < prism_conn.size(); ++i) {
    const Scene::FieldSlice* fs = first_slice_ptr(prism_slices[i]);
    const float* cptr = (fs && fs->offset >=0 && static_cast<size_t>(fs->offset + fs->count) <= coeffs.size())
                        ? coeffs.data() + fs->offset : nullptr;
    uint3 fm = modes;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes3; fcnt = fs->count; }
    zidingyi::PrismElementData p{};
    for (int v = 0; v < 6; ++v) p.vertices[v] = fetch_vertex(prism_conn[i].verts[v]);
    p.fieldModes = fm;
    std::vector<float> gx, gy, gz;
    uint3 geomModes = make_uint3(0,0,0);
    uint3 baseGeom = modes;
    auto itBase = prism_mode_map.find(prism_conn[i].id);
    if (itBase != prism_mode_map.end()) baseGeom = itBase->second;
    build_prism_geometry(prism_conn[i], baseGeom, gx, gy, gz, geomModes);
    scene.add_prism(p, cptr, fm, fcnt,
                    gx.empty()?nullptr:gx.data(),
                    gy.empty()?nullptr:gy.data(),
                    gz.empty()?nullptr:gz.data(),
                    geomModes,
                    gx.size(),
                    &prism_slices[i]);
  }
  for (size_t i = 0; i < quad_conn.size(); ++i) {
    const Scene::FieldSlice* fs = first_slice_ptr(quad_slices[i]);
    const float* cptr = (fs && fs->offset >=0 && static_cast<size_t>(fs->offset + fs->count) <= coeffs.size())
                        ? coeffs.data() + fs->offset : nullptr;
    uint2 fm = quadModes2;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes2; fcnt = fs->count; }
    zidingyi::QuadElementData q{};
    for (int v = 0; v < 4; ++v) q.vertices[v] = fetch_vertex(quad_conn[i].verts[v]);
    std::vector<float> gx, gy, gz;
    uint2 geomModes = make_uint2(0,0);
    uint2 baseGeom = quadModes2;
    auto itBase = quad_mode_map.find(quad_conn[i].id);
    if (itBase != quad_mode_map.end()) baseGeom = itBase->second;
    build_quad_geometry(quad_conn[i], baseGeom, gx, gy, gz, geomModes);
    scene.add_quad(q, cptr, fm, fcnt,
                   gx.empty()?nullptr:gx.data(),
                   gy.empty()?nullptr:gy.data(),
                   gz.empty()?nullptr:gz.data(),
                   geomModes,
                   gx.size(),
                   &quad_slices[i]);
  }
  for (size_t i = 0; i < tri_conn.size(); ++i) {
    const Scene::FieldSlice* fs = first_slice_ptr(tri_slices[i]);
    const float* cptr = (fs && fs->offset >=0 && static_cast<size_t>(fs->offset + fs->count) <= coeffs.size())
                        ? coeffs.data() + fs->offset : nullptr;
    uint2 fm = triModes2;
    size_t fcnt = 0;
    if (fs) { fm = fs->modes2; fcnt = fs->count; }
    zidingyi::TriElementData t{};
    for (int v = 0; v < 3; ++v) t.vertices[v] = fetch_vertex(tri_conn[i].verts[v]);
    t.fieldModes = fm;
    std::vector<float> gx, gy, gz;
    uint2 geomModes = make_uint2(0,0);
    uint2 baseGeom = triModes2;
    auto itBase = tri_mode_map.find(tri_conn[i].id);
    if (itBase != tri_mode_map.end()) baseGeom = itBase->second;
    build_tri_geometry(tri_conn[i], baseGeom, gx, gy, gz, geomModes);
    scene.add_curved_tri(t, cptr, fm, fcnt,
                         gx.empty()?nullptr:gx.data(),
                         gy.empty()?nullptr:gy.data(),
                         gz.empty()?nullptr:gz.data(),
                         geomModes,
                         gx.size(),
                         &tri_slices[i]);
  }

  const size_t coeffs_used = coeff_base;
  if (!coeffs.empty() && coeffs_used < coeffs.size()) {
    std::cout << "XML/FLD loader: unused coeffs = " << (coeffs.size() - coeffs_used) << std::endl;
  } else if (!coeffs.empty() && coeffs_required > coeffs.size()) {
    std::cout << "XML/FLD loader: coefficients appear insufficient, expected at least " << coeffs_required << " values." << std::endl;
  }

  std::cout << "XML/FLD loader: loaded "
            << scene.hexes.size() << " hexes, "
            << scene.prisms.size() << " prisms, "
            << scene.quads.size() << " quads, "
            << scene.curved_tris.size() << " curved tris." << std::endl;
  return true;
}
 
