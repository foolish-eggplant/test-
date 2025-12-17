#include "xml_fld_common.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

NodeDistribution ParseNodeType(const std::string& typeAttr)
{
  std::string up = typeAttr;
  std::transform(up.begin(), up.end(), up.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  if (up.find("EVENLY") != std::string::npos || up.find("POLY") != std::string::npos ||
      up.find("UNIFORM") != std::string::npos)
    return NodeDistribution::EvenlySpaced;
  if (up.find("GLL") != std::string::npos || up.find("GAUSSLOBATTO") != std::string::npos)
    return NodeDistribution::GLL;
  return NodeDistribution::GLL;
}

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
