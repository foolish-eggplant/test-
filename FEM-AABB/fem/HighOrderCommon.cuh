#pragma once

#include <cuda_runtime.h>

namespace zidingyi
{
struct DeviceHighOrderScene
{
  const float4* vertexBuffer{nullptr};
  const int* coordOffsets{nullptr};

  const float* curvedGeomBuffer{nullptr};
  const int* curvedGeomOffset{nullptr};
  const uint3* curvedGeomNumModes{nullptr};

  const float* solutionCoefficients{nullptr};
  const int* coeffOffsets{nullptr};
  const uint3* fieldModes{nullptr};

  int elementCount{0};
};

__device__ __forceinline__ const float4& GetVertex(const DeviceHighOrderScene& scene,
                                                   int elementId,
                                                   int vertexId)
{
  const int base = scene.coordOffsets[elementId];
  return scene.vertexBuffer[base + vertexId];
}

__device__ __forceinline__ const float* GetCurvedGeom(const DeviceHighOrderScene& scene,
                                                      int elementId,
                                                      int component)
{
  const int offset = scene.curvedGeomOffset[elementId];
  if (offset < 0) return nullptr;
  const uint3 modes = scene.curvedGeomNumModes[elementId];
  const int nm = modes.x * modes.y * modes.z;
  return &scene.curvedGeomBuffer[offset + component * nm];
}

__device__ __forceinline__ const float* GetFieldCoefficients(const DeviceHighOrderScene& scene,
                                                            int elementId)
{
  const int offset = scene.coeffOffsets[elementId];
  return &scene.solutionCoefficients[offset];
}

__device__ __forceinline__ uint3 GetFieldModes(const DeviceHighOrderScene& scene,
                                               int elementId)
{
  return scene.fieldModes[elementId];
}
} // namespace zidingyi
