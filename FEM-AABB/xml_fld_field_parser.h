#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "xml_fld_common.h"

struct ExpansionSpec
{
  std::vector<std::pair<char, int>> elems;
  uint3 modes{make_uint3(0, 0, 0)};
  std::vector<std::string> fields;
};

struct FieldData
{
  std::vector<std::string> fieldNames;
  std::vector<float> coeffs;
};

FieldData ParseFldFile(const std::string& fldText);
std::unordered_map<int, std::vector<std::pair<char, int>>> ParseComposites(const std::string& xmlText);
std::vector<ExpansionSpec> ParseExpansions(const std::string& xmlText,
                                           const uint3& defaultModes,
                                           const std::vector<std::string>& fieldNames,
                                           const std::unordered_map<int, std::vector<std::pair<char, int>>>& composites);
