#pragma once

#include <array>
#include <vector>

#include <cuda_runtime.h>

#include "fem/ModalBasis.cuh"
#include "xml_fld_common.h"

std::vector<float> GLLNodes(unsigned int n);
std::vector<float> EvenlySpacedNodes(unsigned int n);
std::vector<float> MakeNodes(unsigned int n, NodeDistribution dist);

bool InvertSquareMatrix(std::vector<float> mat, int n, std::vector<float>& inv);
bool InvertSquareMatrixDouble(std::vector<double> mat, int n, std::vector<double>& inv);

std::vector<float> BuildLegendreInterpMatrix(unsigned int nSrc,
                                             unsigned int nTgt,
                                             NodeDistribution srcType = NodeDistribution::EvenlySpaced,
                                             NodeDistribution tgtType = NodeDistribution::GLL);

float EvalModal1D(const std::vector<float>& coeffs, float r);
float EvalModal2D(const std::vector<float>& coeffs, uint2 modes, float r, float s);
float EvalModal3D(const std::vector<float>& coeffs, uint3 modes, float r, float s, float t);

bool TensorModalFit2D(const std::vector<float3>& nodal,
                      const std::vector<float>& rNodes,
                      const std::vector<float>& sNodes,
                      std::vector<float>& coeffX,
                      std::vector<float>& coeffY,
                      std::vector<float>& coeffZ);
bool TensorModalFit3D(const std::vector<float3>& nodal,
                      const std::vector<float>& rNodes,
                      const std::vector<float>& sNodes,
                      const std::vector<float>& tNodes,
                      std::vector<float>& coeffX,
                      std::vector<float>& coeffY,
                      std::vector<float>& coeffZ);
bool SolveTriModalCoeffs(const std::vector<float3>& nodal,
                         const std::vector<float2>& rsNodes,
                         uint2 modes,
                         std::vector<float>& coeffX,
                         std::vector<float>& coeffY,
                         std::vector<float>& coeffZ);

std::vector<float3> ResampleEdgePoints(const std::vector<float3>& src,
                                       unsigned int targetCount,
                                       bool reverseDirection = false,
                                       NodeDistribution srcType = NodeDistribution::GLL,
                                       NodeDistribution targetType = NodeDistribution::GLL);

std::vector<float3> ResampleQuadFaceLegendre(const std::vector<float3>& src,
                                             unsigned int targetDimR,
                                             unsigned int targetDimS,
                                             const std::array<float3, 4>& expectedCorners,
                                             NodeDistribution srcType = NodeDistribution::EvenlySpaced,
                                             NodeDistribution targetType = NodeDistribution::GLL);
std::vector<float3> ResampleQuadFace(const std::vector<float3>& src,
                                     unsigned int targetDimR,
                                     unsigned int targetDimS,
                                     const std::array<float3, 4>& expectedCorners,
                                     NodeDistribution srcType = NodeDistribution::GLL,
                                     NodeDistribution targetType = NodeDistribution::GLL);
std::vector<float3> ResampleTriFace(const std::vector<float3>& src,
                                    unsigned int targetOrder,
                                    const std::array<float3, 3>& expectedCorners,
                                    NodeDistribution srcType = NodeDistribution::EvenlySpaced,
                                    NodeDistribution targetType = NodeDistribution::GLL);
