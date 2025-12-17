#include "xml_fld_field_parser.h"

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <regex>
#include <sstream>

#include "xml_fld_common.h"

#define XML_FLD_HAS_ZLIB 1
#include <zlib.h>

namespace
{
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
#if !XML_FLD_HAS_ZLIB
  return false;
#else
  output.clear();
  auto try_decompress = [&](int windowBits) -> bool
  {
    z_stream strm{};
    strm.next_in = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(input.data()));
    strm.avail_in = static_cast<uInt>(input.size());
    if (inflateInit2(&strm, windowBits) != Z_OK) return false;

    std::vector<uint8_t> buffer(128 * 1024);

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
#endif
}

std::vector<float> DecodeBinaryCoefficients(const std::vector<uint8_t>& bytes)
{
  std::vector<float> result;
  if (bytes.empty()) return result;

  if (bytes.size() % sizeof(double) != 0 && bytes.size() % sizeof(float) != 0)
  {
    std::cerr << "[Warning] Unexpected coefficient byte size: " << bytes.size() << std::endl;
  }

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
} // namespace

FieldData ParseFldFile(const std::string& fldText)
{
  FieldData data;
  if (fldText.empty()) return data;

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
      if (!name.empty()) data.fieldNames.push_back(name);
    }
  }

  const std::string payload = ExtractElementsPayload(fldText);
  if (!payload.empty())
  {
    ParseAsciiFloats(payload, data.coeffs);
    if (data.coeffs.empty())
    {
      const std::vector<uint8_t> decoded = Base64Decode(payload);
      std::vector<uint8_t> inflated;
      if (!decoded.empty())
      {
        if (!DecompressZlib(decoded, inflated)) inflated = decoded;
        if (!inflated.empty()) data.coeffs = DecodeBinaryCoefficients(inflated);
      }
    }
  }
  return data;
}

std::unordered_map<int, std::vector<std::pair<char, int>>> ParseComposites(const std::string& xmlText)
{
  std::unordered_map<int, std::vector<std::pair<char, int>>> composites;
  std::regex comp_re(R"(<C\s+ID=\"?(\d+)\"?\s*>\s*([^<]+)\s*</C>)", std::regex::icase);
  auto begin = std::sregex_iterator(xmlText.begin(), xmlText.end(), comp_re);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it)
  {
    int cid = std::atoi((*it)[1].str().c_str());
    std::string body = (*it)[2].str();
    std::regex entry_re(R"(([HQPT])\s*\[\s*([0-9\-\:, ]+)\s*\])", std::regex::icase);
    auto b2 = std::sregex_iterator(body.begin(), body.end(), entry_re);
    for (auto jt = b2; jt != end; ++jt)
    {
      char typ = static_cast<char>(std::toupper((*jt)[1].str()[0]));
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
        for (int v = lo; v <= hi; ++v)
        {
          composites[cid].push_back({typ, v});
        }
      }
    }
  }
  return composites;
}

std::vector<ExpansionSpec> ParseExpansions(const std::string& xmlText,
                                           const uint3& defaultModes,
                                           const std::vector<std::string>& fieldNames,
                                           const std::unordered_map<int, std::vector<std::pair<char, int>>>& composites)
{
  std::vector<ExpansionSpec> expansion_specs;
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
                                         : (!numModesStr.empty() ? ParseModes(numModesStr) : defaultModes);
    if (!fieldsAttr.empty()) spec.fields = ParseFieldList(fieldsAttr);

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
          spec.elems.insert(spec.elems.end(), itc->second.begin(), itc->second.end());
        }
      }
    }
    if (!spec.elems.empty()) expansion_specs.push_back(spec);
  }

  return expansion_specs;
}
