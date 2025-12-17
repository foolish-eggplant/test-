#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace zidingyi
{
__host__ __device__ inline float IntegerPow(float base, unsigned int power)
{
  if (power == 0) return 1.0f;
  float result = 1.0f;
  for (unsigned int i = 0; i < power; ++i) result *= base;
  return result;
}

__host__ __device__ inline float JacobiP(int n, float alpha, float beta, float x)
{
  if (n == 0) return 1.0f;
  if (n == 1) return 0.5f * ((2.0f + alpha + beta) * x + (alpha - beta));

  float pk_minus2 = 1.0f;
  float pk_minus1 = 0.5f * ((2.0f + alpha + beta) * x + (alpha - beta));
  float pk = 0.0f;

  for (int k = 2; k <= n; ++k)
  {
    const float kf = static_cast<float>(k);
    const float two_k_ab = 2.0f * kf + alpha + beta;
    const float denom = 2.0f * kf * (kf + alpha + beta) * (two_k_ab - 2.0f);
    const float term1 =
      (two_k_ab - 1.0f) *
      (two_k_ab * (two_k_ab - 2.0f) * x + (alpha * alpha - beta * beta)) *
      pk_minus1;
    const float term2 =
      2.0f * (kf + alpha - 1.0f) * (kf + beta - 1.0f) * two_k_ab * pk_minus2;
    pk = (term1 - term2) / denom;
    pk_minus2 = pk_minus1;
    pk_minus1 = pk;
  }
  return pk;
}

__host__ __device__ inline float JacobiPDerivative(int n,
                                                   float alpha,
                                                   float beta,
                                                   float x)
{
  if (n == 0) return 0.0f;
  const float factor = 0.5f * (n + alpha + beta + 1.0f);
  return factor * JacobiP(n - 1, alpha + 1.0f, beta + 1.0f, x);
}

__host__ __device__ inline float ModifiedA(unsigned int i, float x)
{
  if (i == 0) return 0.5f * (1.0f - x);
  if (i == 1) return 0.5f * (1.0f + x);
  const unsigned int n = i - 2;
  const float poly = JacobiP(static_cast<int>(n), 1.0f, 1.0f, x);
  return 0.25f * (1.0f - x * x) * poly;
}

__host__ __device__ inline float ModifiedAPrime(unsigned int i, float x)
{
  if (i == 0) return -0.5f;
  if (i == 1) return 0.5f;
  const unsigned int n = i - 2;
  const float poly = JacobiP(static_cast<int>(n), 1.0f, 1.0f, x);
  const float dpoly = JacobiPDerivative(static_cast<int>(n), 1.0f, 1.0f, x);
  return -0.5f * x * poly + 0.25f * (1.0f - x * x) * dpoly;
}

__host__ __device__ inline float ModifiedB(unsigned int i,
                                           unsigned int j,
                                           float x)
{
  if (i == 0) return ModifiedA(j, x);
  if (j == 0) return IntegerPow((1.0f - x) * 0.5f, i);
  float result = IntegerPow((1.0f - x) * 0.5f, i);
  result *= 0.5f * (1.0f + x);
  result *= JacobiP(static_cast<int>(j) - 1, 2.0f * i - 1.0f, 1.0f, x);
  return result;
}

__host__ __device__ inline float ModifiedBPrime(unsigned int i,
                                                unsigned int j,
                                                float x)
{
  if (i == 0) return ModifiedAPrime(j, x);
  if (j == 0)
  {
    if (i == 0) return 0.0f;
    float result = -static_cast<float>(i);
    result *= IntegerPow(1.0f - x, i - 1);
    result *= IntegerPow(0.5f, static_cast<int>(i));
    return result;
  }
  const float scale = IntegerPow(0.5f, i + 1);
  const float poly = JacobiP(static_cast<int>(j) - 1, 2.0f * i - 1.0f, 1.0f, x);
  const float dpoly =
    JacobiPDerivative(static_cast<int>(j) - 1, 2.0f * i - 1.0f, 1.0f, x);
  const float term1 = IntegerPow(1.0f - x, i) * poly;
  const float term2 =
    static_cast<float>(i) * IntegerPow(1.0f - x, i - 1) * (1.0f + x) * poly;
  const float term3 =
    IntegerPow(1.0f - x, i) * (1.0f + x) * dpoly;
  return scale * (term1 - term2 + term3);
}
} // namespace zidingyi
