#include "xml_fld_math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <cstdio>

namespace
{
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
    const double k_d = static_cast<double>(k);
    const double pk = ((2.0 * k_d - 1.0) * x * p1 - (k_d - 1.0) * p0) / k_d;
    p0 = p1;
    p1 = pk;
  }
  pnm1 = p0;
  return p1;
}

inline double LegendrePDer(int n, double x, double pn, double pnm1)
{
  const double denom = x * x - 1.0;
  if (std::fabs(denom) < 1e-14) return 0.0;
  return (static_cast<double>(n) * (x * pn - pnm1)) / denom;
}

inline double LegendrePValue(int n, double x)
{
  double prev = 0.0;
  return LegendreP(n, x, prev);
}

float CornerDistanceSq(const float3& a, const float3& b)
{
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}
} // namespace

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

  const unsigned int N = n - 1;
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
  if (n == 1)
  {
    nodes.push_back(0.0f);
    return nodes;
  }
  nodes.resize(n);
  const float step = 2.0f / static_cast<float>(n - 1);
  for (unsigned int i = 0; i < n; ++i)
  {
    nodes[i] = -1.0f + step * static_cast<float>(i);
  }
  return nodes;
}

std::vector<float> MakeNodes(unsigned int n, NodeDistribution dist)
{
  return (dist == NodeDistribution::EvenlySpaced) ? EvenlySpacedNodes(n) : GLLNodes(n);
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
      if (v > maxAbs)
      {
        maxAbs = v;
        pivot = r;
      }
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
    for (int c = 0; c < n; ++c)
    {
      mat[col * n + c] *= invPivot;
      inv[col * n + c] *= invPivot;
    }
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
                                             NodeDistribution srcType,
                                             NodeDistribution tgtType)
{
  if (nSrc == 0 || nTgt == 0) return {};
  const std::vector<float> srcNodes = MakeNodes(nSrc, srcType);
  const std::vector<float> tgtNodes = MakeNodes(nTgt, tgtType);

  std::vector<float> Vsrc(nSrc * nSrc);
  for (unsigned int i = 0; i < nSrc; ++i)
    for (unsigned int j = 0; j < nSrc; ++j)
      Vsrc[i * nSrc + j] = static_cast<float>(LegendrePValue(static_cast<int>(j), srcNodes[i]));

  std::vector<float> VsrcInv;
  if (!InvertSquareMatrix(Vsrc, static_cast<int>(nSrc), VsrcInv)) return {};

  std::vector<float> interp(nTgt * nSrc, 0.0f);
  for (unsigned int i = 0; i < nTgt; ++i)
  {
    for (unsigned int j = 0; j < nSrc; ++j)
    {
      double sum = 0.0;
      for (unsigned int k = 0; k < nSrc; ++k)
      {
        const float basis = static_cast<float>(LegendrePValue(static_cast<int>(k), tgtNodes[i]));
        sum += basis * VsrcInv[k * nSrc + j];
      }
      interp[i * nSrc + j] = static_cast<float>(sum);
    }
  }
  return interp;
}

float EvalModal1D(const std::vector<float>& coeffs, float r)
{
  float val = 0.0f;
  for (size_t i = 0; i < coeffs.size(); ++i)
  {
    val += coeffs[i] * zidingyi::ModifiedA(static_cast<unsigned int>(i), r);
  }
  return val;
}

float EvalModal2D(const std::vector<float>& coeffs, uint2 modes, float r, float s)
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

float EvalModal3D(const std::vector<float>& coeffs, uint3 modes, float r, float s, float t)
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

  std::vector<double> Vr(nr * nr), Vs(ns * ns), Vr_inv, Vs_inv;
  for (int i = 0; i < nr; ++i)
    for (int a = 0; a < nr; ++a)
      Vr[i * nr + a] = static_cast<double>(zidingyi::ModifiedA(a, rNodes[i]));
  for (int j = 0; j < ns; ++j)
    for (int b = 0; b < ns; ++b)
      Vs[j * ns + b] = static_cast<double>(zidingyi::ModifiedA(b, sNodes[j]));
  if (!InvertSquareMatrixDouble(Vr, nr, Vr_inv) || !InvertSquareMatrixDouble(Vs, ns, Vs_inv)) return false;

  std::vector<double> tempX(nr * ns, 0.0), tempY(nr * ns, 0.0), tempZ(nr * ns, 0.0);
  for (int j = 0; j < ns; ++j)
  {
    for (int a = 0; a < nr; ++a)
    {
      double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
      for (int i = 0; i < nr; ++i)
      {
        const double w = Vr_inv[a * nr + i];
        const float3 v = nodal[j * nr + i];
        sumX += w * v.x;
        sumY += w * v.y;
        sumZ += w * v.z;
      }
      const size_t idx = static_cast<size_t>(j) * nr + a;
      tempX[idx] = sumX;
      tempY[idx] = sumY;
      tempZ[idx] = sumZ;
    }
  }

  coeffX.assign(nr * ns, 0.0f);
  coeffY.assign(nr * ns, 0.0f);
  coeffZ.assign(nr * ns, 0.0f);
  for (int b = 0; b < ns; ++b)
  {
    for (int a = 0; a < nr; ++a)
    {
      double sx = 0.0, sy = 0.0, sz = 0.0;
      for (int j = 0; j < ns; ++j)
      {
        const double w = Vs_inv[b * ns + j];
        const size_t idx = static_cast<size_t>(j) * nr + a;
        sx += w * tempX[idx];
        sy += w * tempY[idx];
        sz += w * tempZ[idx];
      }
      const size_t idx = static_cast<size_t>(b) * nr + a;
      coeffX[idx] = static_cast<float>(sx);
      coeffY[idx] = static_cast<float>(sy);
      coeffZ[idx] = static_cast<float>(sz);
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
          sx += w * v.x;
          sy += w * v.y;
          sz += w * v.z;
        }
        const size_t idx = static_cast<size_t>(k * ns + j) * nr + a;
        tempX_r[idx] = sx;
        tempY_r[idx] = sy;
        tempZ_r[idx] = sz;
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
        tempX_rs[idxOut] = sx;
        tempY_rs[idxOut] = sy;
        tempZ_rs[idxOut] = sz;
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
                                       bool reverseDirection,
                                       NodeDistribution srcType,
                                       NodeDistribution targetType)
{
  if (src.empty() || targetCount == 0) return {};
  const unsigned int nSrc = static_cast<unsigned int>(src.size());
  const std::vector<float> rSrc = MakeNodes(nSrc, srcType);
  std::vector<double> Vr(nSrc * nSrc), Vr_inv;
  for (unsigned int i = 0; i < nSrc; ++i)
    for (unsigned int a = 0; a < nSrc; ++a)
      Vr[i * nSrc + a] = static_cast<double>(zidingyi::ModifiedA(a, rSrc[i]));
  if (!InvertSquareMatrixDouble(Vr, static_cast<int>(nSrc), Vr_inv)) return {};

  std::vector<double> coeffX(nSrc, 0.0), coeffY(nSrc, 0.0), coeffZ(nSrc, 0.0);
  for (unsigned int a = 0; a < nSrc; ++a)
  {
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (unsigned int i = 0; i < nSrc; ++i)
    {
      const double w = Vr_inv[a * nSrc + i];
      const float3 v = src[i];
      sx += w * v.x;
      sy += w * v.y;
      sz += w * v.z;
    }
    coeffX[a] = sx;
    coeffY[a] = sy;
    coeffZ[a] = sz;
  }

  const std::vector<float> rT = MakeNodes(targetCount, targetType);
  std::vector<float3> out(targetCount);
  for (unsigned int i = 0; i < targetCount; ++i)
  {
    const float r = reverseDirection ? rT[targetCount - 1 - i] : rT[i];
    double vx = 0.0, vy = 0.0, vz = 0.0;
    for (size_t a = 0; a < coeffX.size(); ++a)
    {
      const double basis = static_cast<double>(zidingyi::ModifiedA(static_cast<unsigned int>(a), r));
      vx += coeffX[a] * basis;
      vy += coeffY[a] * basis;
      vz += coeffZ[a] * basis;
    }
    out[i] = make_float3(static_cast<float>(vx),
                         static_cast<float>(vy),
                         static_cast<float>(vz));
  }
  return out;
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
  const float tol = 1e-3f;
  bool matched = false;
  std::vector<float3> candidate(grid.size());
  float minErr = std::numeric_limits<float>::max();
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
        float err = CornerDistanceSq(c[0], desiredCorners[0]) +
                    CornerDistanceSq(c[1], desiredCorners[1]) +
                    CornerDistanceSq(c[2], desiredCorners[2]) +
                    CornerDistanceSq(c[3], desiredCorners[3]);
        if (err < minErr) minErr = err;
        if (err < tol)
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
  else
  {
    std::printf("Warning: ReorderQuadGrid failed (dim=%u, minErr=%f)\n", dim, minErr);
    std::printf("  Desired Corners:\n");
    for (int i = 0; i < 4; ++i)
      std::printf("    [%d] %.4f %.4f %.4f\n", i, desiredCorners[i].x, desiredCorners[i].y, desiredCorners[i].z);
    std::printf("  Grid Corners (Raw):\n");
    float3 g0 = fetch(0, 0);
    float3 g1 = fetch(dim - 1, 0);
    float3 g2 = fetch(dim - 1, dim - 1);
    float3 g3 = fetch(0, dim - 1);
    std::printf("    (0,0)     %.4f %.4f %.4f\n", g0.x, g0.y, g0.z);
    std::printf("    (max,0)   %.4f %.4f %.4f\n", g1.x, g1.y, g1.z);
    std::printf("    (max,max) %.4f %.4f %.4f\n", g2.x, g2.y, g2.z);
    std::printf("    (0,max)   %.4f %.4f %.4f\n", g3.x, g3.y, g3.z);
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

std::vector<float3> ResampleQuadFaceLegendre(const std::vector<float3>& src,
                                             unsigned int targetDimR,
                                             unsigned int targetDimS,
                                             const std::array<float3, 4>& expectedCorners,
                                             NodeDistribution srcType,
                                             NodeDistribution targetType)
{
  const size_t srcSize = src.size();
  const unsigned int srcDim = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(srcSize))));
  if (srcDim * srcDim != srcSize || targetDimR == 0 || targetDimS == 0) return {};

  std::vector<float3> oriented = src;
  ReorderQuadGrid(src, srcDim, expectedCorners, oriented);

  const std::vector<float> rowMat = BuildLegendreInterpMatrix(srcDim, targetDimR, srcType, targetType);
  const std::vector<float> colMat = BuildLegendreInterpMatrix(srcDim, targetDimS, srcType, targetType);
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
        sx += w * p.x;
        sy += w * p.y;
        sz += w * p.z;
      }
      temp[j * targetDimR + i] = make_float3(static_cast<float>(sx),
                                             static_cast<float>(sy),
                                             static_cast<float>(sz));
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
        sx += w * p.x;
        sy += w * p.y;
        sz += w * p.z;
      }
      out[j * targetDimR + i] = make_float3(static_cast<float>(sx),
                                            static_cast<float>(sy),
                                            static_cast<float>(sz));
    }
  }
  return out;
}

std::vector<float3> ResampleQuadFace(const std::vector<float3>& src,
                                     unsigned int targetDimR,
                                     unsigned int targetDimS,
                                     const std::array<float3, 4>& expectedCorners,
                                     NodeDistribution srcType,
                                     NodeDistribution targetType)
{
  const size_t srcSize = src.size();
  const unsigned int srcDim = static_cast<unsigned int>(std::round(std::sqrt(static_cast<double>(srcSize))));
  if (srcDim * srcDim != srcSize || targetDimR == 0 || targetDimS == 0) return {};
  std::vector<float3> oriented = src;
  ReorderQuadGrid(src, srcDim, expectedCorners, oriented);

  const std::vector<float> rSrc = MakeNodes(srcDim, srcType);
  const std::vector<float> sSrc = MakeNodes(srcDim, srcType);
  std::vector<float> coeffX, coeffY, coeffZ;
  if (!TensorModalFit2D(oriented, rSrc, sSrc, coeffX, coeffY, coeffZ)) return {};

  const std::vector<float> rT = MakeNodes(targetDimR, targetType);
  const std::vector<float> sT = MakeNodes(targetDimS, targetType);
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
                                    const std::array<float3, 3>& expectedCorners,
                                    NodeDistribution srcType,
                                    NodeDistribution targetType)
{
  const size_t expectedSrc = static_cast<size_t>(targetOrder) * (targetOrder + 1) / 2;
  if (src.size() < expectedSrc || targetOrder < 2) return {};

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
      float r = 2.0f * u - 1.0f;
      float s = 2.0f * v - 1.0f;
      if (srcType == NodeDistribution::GLL)
      {
        const std::vector<float> nodes1D = GLLNodes(srcOrder);
        r = nodes1D[i];
        s = nodes1D[j];
      }
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
  const std::vector<float> r1dT = MakeNodes(tgt, targetType);
  for (unsigned int j = 0; j < tgt; ++j)
  {
    for (unsigned int i = 0; i + j < tgt; ++i)
    {
      float r = 2.0f * static_cast<float>(i) / static_cast<float>(tgt - 1) - 1.0f;
      float s = 2.0f * static_cast<float>(j) / static_cast<float>(tgt - 1) - 1.0f;
      if (targetType == NodeDistribution::GLL)
      {
        r = r1dT[i];
        s = r1dT[j];
      }
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
