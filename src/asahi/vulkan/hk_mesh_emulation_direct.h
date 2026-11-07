/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"
#include "asahi/libagx/libagx_dgc.h"
#include "hk_nir_lower_mesh_emulation.h"

struct hk_api_shader;
struct hk_shader;

/*
 * Non-public command contract for the first direct G13 software-mesh draw.
 * It deliberately stops before command recording: a future caller must put
 * producer_grid on the pre-graphics compute stream, retain the graphics root
 * for producer, issue barrier, then consume draw through the ordinary raster
 * path. The command allocation must be zero-initialized: the producer writes
 * the first 16 bytes of the group header and draw command, including an
 * explicit zero indexCount on ABI mismatch; reserved tails stay zero.
 * Keeping these values together prevents an entry point from silently
 * substituting an uncompiled NIR helper or a non-U32 draw.
 */
struct hk_mesh_emulation_direct_plan {
   const struct hk_shader *producer;
   const struct hk_shader *raster;
   struct hk_mesh_emulation_params params;
   struct agx_grid producer_grid;
   struct agx_workgroup producer_local_size;
   enum agx_barrier barrier;
   struct agx_draw draw;
   uint64_t vertex_records_size_B;
   uint64_t index_buffer_size_B;
   uint64_t group_records_size_B;
};

bool hk_mesh_emulation_direct_plan_init(
   enum agx_chip chip, const struct hk_api_shader *api,
   const uint32_t group_count[3], uint64_t vertex_records,
   uint64_t vertex_records_size_B, uint64_t index_buffer,
   uint64_t index_buffer_size_B, uint64_t group_records,
   uint64_t group_records_size_B, uint64_t draw_params,
   struct hk_mesh_emulation_direct_plan *plan);
