#include "vtu_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <type_traits>

// 定义 VTK 单元类型常量
namespace VTKCellType {
    // 线性单元
    const int TRIANGLE = 5;
    const int QUAD = 9;
    const int TETRA = 10;
    const int HEXAHEDRON = 12;
    const int WEDGE = 13; // Prism

    // Lagrange 高阶单元
    const int LAGRANGE_TRIANGLE = 69;
    const int LAGRANGE_QUADRILATERAL = 70;
    const int LAGRANGE_TETRAHEDRON = 71;
    const int LAGRANGE_HEXAHEDRON = 72;
    const int LAGRANGE_WEDGE = 73;

    // Bezier 高阶单元 (节点顺序通常也是角点在前)
    const int BEZIER_TRIANGLE = 21;
    const int BEZIER_QUADRILATERAL = 22;
    const int BEZIER_TETRAHEDRON = 24;
    const int BEZIER_HEXAHEDRON = 25;
    const int BEZIER_WEDGE = 26;
}

namespace
{

// 辅助：读取整个文件到字符串
std::string ReadFileToString(const std::string& path)
{
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.is_open()) return {};
    
    in.seekg(0, std::ios::end);
    size_t fileSize = in.tellg();
    if (fileSize == 0) return {};
    in.seekg(0, std::ios::beg);

    std::string str(fileSize, '\0');
    in.read(&str[0], fileSize);
    return str;
}

// 辅助：简单的 XML 标签查找器
std::string GetTagContent(const std::string& xmlString, const std::string& tagName, size_t& outEndPos, size_t startPos = 0)
{
    std::string openTag = "<" + tagName;
    std::string closeTag = "</" + tagName + ">";

    size_t start = xmlString.find(openTag, startPos);
    if (start == std::string::npos) return "";

    // 找到 > 结束符
    size_t contentStart = xmlString.find('>', start);
    if (contentStart == std::string::npos) return "";
    contentStart += 1; 

    size_t end = xmlString.find(closeTag, contentStart);
    if (end == std::string::npos) return "";

    outEndPos = end + closeTag.length();
    return xmlString.substr(contentStart, end - contentStart);
}

// 辅助：提取属性值
std::string GetAttributeValue(const std::string& tagHeader, const std::string& attrName)
{
    size_t pos = tagHeader.find(attrName + "=");
    if (pos == std::string::npos) return "";

    char quote = tagHeader[pos + attrName.length() + 1]; 
    if (quote != '\"' && quote != '\'') return "";

    size_t start = pos + attrName.length() + 2;
    size_t end = tagHeader.find(quote, start);
    if (end == std::string::npos) return "";

    return tagHeader.substr(start, end - start);
}

// 核心解析函数：提取 DataArray
template <typename T>
bool ExtractDataArray(const std::string& parentContent, 
                      const std::string& nameMatch, 
                      std::vector<T>& outValues)
{
    size_t searchPos = 0;
    while (true)
    {
        size_t tagStart = parentContent.find("<DataArray", searchPos);
        if (tagStart == std::string::npos) break;

        size_t headerEnd = parentContent.find('>', tagStart);
        if (headerEnd == std::string::npos) break;

        std::string header = parentContent.substr(tagStart, headerEnd - tagStart);
        
        // 1. 检查 Format
        std::string format = GetAttributeValue(header, "format");
        std::transform(format.begin(), format.end(), format.begin(), [](unsigned char c){ return std::tolower(c); });
        if (!format.empty() && format != "ascii") {
            searchPos = headerEnd + 1;
            continue; 
        }

        // 2. 检查 Name
        if (!nameMatch.empty()) {
            std::string name = GetAttributeValue(header, "Name");
            if (name != nameMatch) {
                searchPos = headerEnd + 1;
                continue;
            }
        }

        // 3. 提取内容
        size_t contentEnd = parentContent.find("</DataArray>", headerEnd);
        if (contentEnd == std::string::npos) break;

        std::string dataStr = parentContent.substr(headerEnd + 1, contentEnd - headerEnd - 1);
        
        // 4. 解析数值
        std::istringstream iss(dataStr);
        outValues.clear();
        outValues.reserve(dataStr.size() / 5); 

        // 特殊处理 char/uint8 类型，避免被当做字符读取
        if (std::is_same<T, uint8_t>::value || std::is_same<T, char>::value || std::is_same<T, unsigned char>::value) {
            int temp;
            while (iss >> temp) {
                outValues.push_back(static_cast<T>(temp));
            }
        } else {
            T val;
            while (iss >> val) {
                outValues.push_back(val);
            }
        }

        return !outValues.empty();
    }
    return false;
}

} // namespace

bool load_vtu_file(const std::string& path, Scene& scene)
{
    std::cout << "VTU loader: Reading " << path << "..." << std::endl;
    const std::string text = ReadFileToString(path);
    if (text.empty())
    {
        std::cerr << "VTU loader: failed to read file or file is empty: " << path << std::endl;
        return false;
    }

    size_t endPos = 0;
    std::string pointsSection = GetTagContent(text, "Points", endPos);
    if (pointsSection.empty()) {
        std::cerr << "VTU loader: <Points> section not found." << std::endl;
        return false;
    }

    std::string cellsSection = GetTagContent(text, "Cells", endPos);
    if (cellsSection.empty()) {
        std::cerr << "VTU loader: <Cells> section not found." << std::endl;
        return false;
    }

    // --- 解析 Points ---
    std::vector<float> pointsRaw;
    if (!ExtractDataArray(pointsSection, "", pointsRaw)) {
        if (!ExtractDataArray(pointsSection, "Points", pointsRaw)) {
            std::cerr << "VTU loader: failed to parse Points data." << std::endl;
            return false;
        }
    }

    if (pointsRaw.size() % 3 != 0) {
        std::cerr << "VTU loader: Points data size error." << std::endl;
        return false;
    }
    size_t numPoints = pointsRaw.size() / 3;

    // --- 解析 Connectivity ---
    std::vector<int> connectivity;
    // 优先尝试读取 int
    if (!ExtractDataArray(cellsSection, "connectivity", connectivity)) {
        // 如果失败，尝试读取 Int64 (long long) 并转换
        std::vector<long long> connectivity64;
        if (ExtractDataArray(cellsSection, "connectivity", connectivity64)) {
             connectivity.resize(connectivity64.size());
             for(size_t i=0; i<connectivity64.size(); ++i) connectivity[i] = static_cast<int>(connectivity64[i]);
        } else {
            std::cerr << "VTU loader: failed to parse Cells connectivity." << std::endl;
            return false;
        }
    }

    // --- 解析 Offsets ---
    std::vector<int> offsets;
    if (!ExtractDataArray(cellsSection, "offsets", offsets)) {
        std::vector<long long> offsets64;
        if (ExtractDataArray(cellsSection, "offsets", offsets64)) {
            offsets.resize(offsets64.size());
            for(size_t i=0; i<offsets64.size(); ++i) offsets[i] = static_cast<int>(offsets64[i]);
        }
    }

    // --- 解析 Types ---
    std::vector<int> types;
    // types 通常是 UInt8
    if (!ExtractDataArray(cellsSection, "types", types)) {
        std::cerr << "VTU loader: failed to parse Cells types." << std::endl;
        return false;
    }

    // Lambda: 安全获取点坐标
    auto getPoint = [&](int idx) -> float3 {
        if (idx < 0 || idx >= (int)numPoints) return make_float3(0,0,0);
        const int base = idx * 3;
        return make_float3(pointsRaw[base + 0], pointsRaw[base + 1], pointsRaw[base + 2]);
    };

    size_t connCursor = 0;
    int loadedCount = 0;

    for (size_t cell = 0; cell < types.size(); ++cell)
    {
        const int type = types[cell];
        int count = 0;

        // 1. 确定当前单元的节点数
        if (cell < offsets.size()) {
            const int prev = (cell == 0) ? 0 : offsets[cell - 1];
            count = offsets[cell] - prev;
        } else {
            // 如果没有 offsets，根据类型猜测 (仅针对标准线性单元)
            if (type == VTKCellType::HEXAHEDRON) count = 8;
            else if (type == VTKCellType::WEDGE) count = 6;
            else if (type == VTKCellType::QUAD) count = 4;
        }

        // 2. 越界检查
        if (count <= 0 || connCursor + count > connectivity.size()) {
            std::cerr << "VTU loader: invalid connectivity range at cell " << cell << std::endl;
            break;
        }

        // 3. 根据类型加载
        // 注意：VTK 标准规定，高阶单元的前 N 个节点就是其角点 (Corner Nodes)。
        // 例如 Lagrange Hex (27点)，前 0-7 号点就是标准的 8 个角点。
        // 因此，我们可以安全地读取前几个点来构建基础几何。

        // --- 六面体 (Hexahedron) ---
        if (type == VTKCellType::HEXAHEDRON || 
            type == VTKCellType::LAGRANGE_HEXAHEDRON || 
            type == VTKCellType::BEZIER_HEXAHEDRON)
        {
            if (count >= 8) {
                zidingyi::HexElementData hex{};
                // 读取前8个角点
                for (int i = 0; i < 8; ++i) hex.vertices[i] = getPoint(connectivity[connCursor + i]);
                
                // TODO: 如果你的 HexElementData 支持高阶系数，在这里读取剩余的点
                // 例如：if (count == 27) { ... load remaining 19 points into hex field/geom coefficients ... }
                hex.fieldCoefficients = nullptr; 
                hex.fieldModes = make_uint3(0, 0, 0);
                
                scene.add_hex(hex);
                loadedCount++;
            }
        }
        // --- 三棱柱 (Prism / Wedge) ---
        else if (type == VTKCellType::WEDGE || 
                 type == VTKCellType::LAGRANGE_WEDGE || 
                 type == VTKCellType::BEZIER_WEDGE)
        {
            if (count >= 6) {
                zidingyi::PrismElementData prism{};
                // 读取前6个角点
                for (int i = 0; i < 6; ++i) prism.vertices[i] = getPoint(connectivity[connCursor + i]);
                
                prism.fieldCoefficients = nullptr;
                prism.geomCoefficients[0] = prism.geomCoefficients[1] = prism.geomCoefficients[2] = nullptr;
                prism.geomModes = make_uint3(0, 0, 0);
                prism.fieldModes = make_uint3(0, 0, 0);
                
                scene.add_prism(prism);
                loadedCount++;
            }
        }
        // --- 四边形 (Quad) ---
        else if (type == VTKCellType::QUAD || 
                 type == VTKCellType::LAGRANGE_QUADRILATERAL || 
                 type == VTKCellType::BEZIER_QUADRILATERAL)
        {
            if (count >= 4) {
                zidingyi::QuadElementData quad{};
                // 读取前4个角点
                for (int i = 0; i < 4; ++i) quad.vertices[i] = getPoint(connectivity[connCursor + i]);
                
                quad.fieldCoefficients = nullptr;
                quad.geomCoefficients[0] = quad.geomCoefficients[1] = quad.geomCoefficients[2] = nullptr;
                quad.fieldModes = make_uint2(0, 0);
                quad.geomModes = make_uint2(0, 0);
                
                scene.add_quad(quad);
                loadedCount++;
            }
        }
        else {
            // 仅在第一次遇到不支持的类型时打印警告，避免刷屏
            static bool warned = false;
            if (!warned) {
                std::cout << "VTU loader: Warning - skipping unsupported cell type ID: " << type << std::endl;
                warned = true;
            }
        }

        connCursor += count;
    }

    if (loadedCount > 0)
    {
        std::cout << "VTU loader: Successfully loaded " << loadedCount << " elements." << std::endl;
        return true;
    }
    
    std::cerr << "VTU loader: No supported elements found in file." << std::endl;
    return false;
}
