#ifndef SCENE_LOADER_H
#define SCENE_LOADER_H

#include "raytracer.h"  // 直接包含 raytracer.h，确保可以访问 Scene 和其他定义

// 声明函数：加载简单场景文件
// 这里的 `Scene` 已经是完整定义了
bool load_scene_file(const std::string& path, Scene& scene);

#endif 