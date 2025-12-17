#pragma once

#include <string>
#include <vector>

#include <cuda_runtime.h>

enum class NodeDistribution
{
  GLL,
  EvenlySpaced
};

NodeDistribution ParseNodeType(const std::string& typeAttr);
std::string ReadFile(const std::string& path);
std::string Trim(const std::string& s);
std::vector<std::string> ParseFieldList(const std::string& s);
std::string GetAttribute(const std::string& tag, const std::string& key);
uint3 ParseModes(const std::string& s);
size_t HexCoeffCount(const uint3& modes);
size_t QuadCoeffCount(const uint2& modes);
size_t TriCoeffCount(const uint2& modes);
size_t PrismCoeffCount(const uint3& modes);
