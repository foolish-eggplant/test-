#pragma once

#include <cuda_runtime.h>

namespace zidingyi
{
__device__ __forceinline__ float IntegerPow(float base, unsigned int power)
{
  if (power == 0) return 1.0f;
  float result = 1.0f;
  for (unsigned int i = 0; i < power; ++i) result *= base;
  return result;
}

__device__ __forceinline__ float ModifiedA(unsigned int i, float x)
{
  if (i == 0) return 0.5f * (1.0f - x);
  if (i == 1) return 0.5f * (1.0f + x);
  float xm = 0.5f * (1.0f - x);
  float xp = 0.5f * (1.0f + x);
  float poly = 1.0f;
  float prev = 1.0f;
  float prevPrev = 0.0f;
  for (unsigned int k = 2; k <= i; ++k)
  {
    if (k == 2)
    {
      poly = 0.5f * (3.0f * x + 1.0f);
    }
    else
    {
      float a = (2.0f * k + 1.0f) / (k + 1.0f);
      float b = k / (k + 1.0f);
      float next = (a * x + b) * prev - b * prevPrev;
      prevPrev = prev;
      prev = next;
      poly = next;
    }
  }
  return xm * xp * poly;
}

__device__ __forceinline__ float ModifiedAPrime(unsigned int i, float x)
{
  if (i == 0) return -0.5f;
  if (i == 1) return 0.5f;
  const float xm = 0.5f * (1.0f - x);
  const float xp = 0.5f * (1.0f + x);
  const float poly = ModifiedA(i - 2, x);
  const float dpoly = 0.5f * (i - 1) * ModifiedA(i - 1, x);
  return -0.5f * xp * poly + 0.5f * xm * poly + xm * xp * dpoly;
}
} // namespace zidingyi
