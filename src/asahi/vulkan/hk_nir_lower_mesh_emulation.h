/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "compiler/nir/nir.h"

/*
 * Driver-internal parameter block for a software mesh producer.  The command
 * path will place its address in draw.geometry_params before dispatching the
 * lowered shader.  Keep this ABI independent of Vulkan application memory.
 */
struct hk_mesh_emulation_params {
   uint64_t vertex_buffer;
   uint64_t index_buffer;
};

struct hk_mesh_emulation_layout {
   unsigned shared_base;
   unsigned vertex_count;
   unsigned primitive_count;
   unsigned position;
   unsigned indices;
   unsigned shared_size;

   unsigned vertices_per_group;
   unsigned primitives_per_group;
};

/*
 * Lower the bounded triangle-only mesh producer contract to a compute shader.
 *
 * The extension remains unadvertised on G13 until the command and generated
 * pull-vertex consumer paths are complete.  Returning false is fail-closed:
 * unsupported shaders are not modified.
 */
bool hk_nir_lower_mesh_emulation(
   nir_shader *shader, struct hk_mesh_emulation_layout *layout_out);
