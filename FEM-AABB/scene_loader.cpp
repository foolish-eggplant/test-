#include "raytracer.h"
#include "scene_loader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

// Supported format (lines, '#' starts a comment):
//   mat_lambert r g b
//   mat_metal r g b roughness
//   mat_dielectric ior
//   mat_emissive r g b  (Lambert base with emission)
//   sphere cx cy cz radius material_id
//   tri x1 y1 z1 x2 y2 z2 x3 y3 z3 material_id
// material_id is zero-based index into the materials list (in order defined).

static bool parse_floats(std::istringstream& iss, float* vals, int n) {
    for (int i = 0; i < n; ++i) {
        if (!(iss >> vals[i])) return false;
    }
    return true;
}

bool load_scene_file(const std::string& path, Scene& scene) {
    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Failed to open scene file: " << path << std::endl;
        return false;
    }

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        // Trim leading spaces
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) continue; // empty line
        if (line[first] == '#') continue; // comment

        std::istringstream iss(line.substr(first));
        std::string cmd;
        if (!(iss >> cmd)) continue;

        if (cmd == "mat_lambert") {
            float vals[3];
            if (!parse_floats(iss, vals, 3)) {
                std::cerr << "[scene] mat_lambert parse error at line " << line_no << std::endl;
                continue;
            }
            scene.add_material(Material(make_float3(vals[0], vals[1], vals[2]), make_float3(0,0,0), 0.0f, 0.0f, 0));
        } else if (cmd == "mat_metal") {
            float vals[4];
            if (!parse_floats(iss, vals, 4)) {
                std::cerr << "[scene] mat_metal parse error at line " << line_no << std::endl;
                continue;
            }
            scene.add_material(Material(make_float3(vals[0], vals[1], vals[2]), make_float3(0,0,0), vals[3], 0.0f, 1));
        } else if (cmd == "mat_dielectric") {
            float ior;
            if (!(iss >> ior)) {
                std::cerr << "[scene] mat_dielectric parse error at line " << line_no << std::endl;
                continue;
            }
            scene.add_material(Material(make_float3(1,1,1), make_float3(0,0,0), 0.0f, ior, 2));
        } else if (cmd == "mat_emissive") {
            float vals[3];
            if (!parse_floats(iss, vals, 3)) {
                std::cerr << "[scene] mat_emissive parse error at line " << line_no << std::endl;
                continue;
            }
            // Use Lambert base with emission
            scene.add_material(Material(make_float3(0,0,0), make_float3(vals[0], vals[1], vals[2]), 0.0f, 0.0f, 0));
        } else if (cmd == "sphere") {
            float vals[5];
            if (!parse_floats(iss, vals, 5)) {
                std::cerr << "[scene] sphere parse error at line " << line_no << std::endl;
                continue;
            }
            int mat_id = static_cast<int>(vals[4]);
            if (mat_id < 0 || mat_id >= static_cast<int>(scene.materials.size())) {
                std::cerr << "[scene] sphere material_id out of range at line " << line_no << std::endl;
                continue;
            }
            scene.add_sphere(Sphere(make_float3(vals[0], vals[1], vals[2]), vals[3], mat_id));
        } else if (cmd == "tri") {
            float vals[10];
            if (!parse_floats(iss, vals, 10)) {
                std::cerr << "[scene] tri parse error at line " << line_no << std::endl;
                continue;
            }
            int mat_id = static_cast<int>(vals[9]);
            if (mat_id < 0 || mat_id >= static_cast<int>(scene.materials.size())) {
                std::cerr << "[scene] tri material_id out of range at line " << line_no << std::endl;
                continue;
            }
            scene.add_triangle(Triangle(
                make_float3(vals[0], vals[1], vals[2]),
                make_float3(vals[3], vals[4], vals[5]),
                make_float3(vals[6], vals[7], vals[8]),
                mat_id
            ));
        } else {
            std::cerr << "[scene] unknown command '" << cmd << "' at line " << line_no << std::endl;
        }
    }

    return true;
}

