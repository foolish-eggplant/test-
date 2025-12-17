#pragma once

#include "raytracer.h"
#include <string>

// Minimal VTU (VTK XML UnstructuredGrid) loader that supports hex/wedge(prism)/quad cells
// with triangular prisms (type 13), hexahedra (type 12), and quads (type 9).
// Geometry is populated; field coefficients are left null (geometry-only intersection).
bool load_vtu_file(const std::string& path, Scene& scene);
