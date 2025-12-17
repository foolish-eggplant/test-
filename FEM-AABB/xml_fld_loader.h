#pragma once

#include "raytracer.h"
#include <string>

// Load a Nektar++-style XML mesh together with an FLD field file.
// - xml_path: mesh/expansion file (typically *.xml)
// - fld_path: field coefficients (typically *.fld)
// Returns true on success and appends elements/materials into the Scene.
// Currently supports hexahedral/prismatic/quad/triangle elements with modified
// Legendre (Modified A/B) modal coefficients. Curved geometry described via
// <CURVED>/<EDGE>/<FACE> blocks is converted into high-order mapping modes
// separate from the field modes.
bool load_xml_fld(const std::string& xml_path,
                  const std::string& fld_path,
                  Scene& scene);
