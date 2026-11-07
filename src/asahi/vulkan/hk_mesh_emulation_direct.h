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

/* Experimental G13 software-mesh profile.  It is intentionally narrower
 * than VK_EXT_mesh_shader Required_Limits and is exposed only by the explicit
 * bring-up option until every wider dimension has submitted-pixel proof. */
#define HK_G13_MESH_MAX_GROUP_TOTAL        4096u
#define HK_G13_MESH_MAX_GROUP_COUNT        65535u
#define HK_G13_MESH_MAX_INVOCATIONS \
   HK_MESH_EMULATION_MAX_WORKGROUP_INVOCATIONS
#define HK_G13_MESH_MAX_WORKGROUP_X HK_MESH_EMULATION_MAX_WORKGROUP_X
#define HK_G13_MESH_MAX_WORKGROUP_Y HK_MESH_EMULATION_MAX_WORKGROUP_Y
#define HK_G13_MESH_MAX_WORKGROUP_Z HK_MESH_EMULATION_MAX_WORKGROUP_Z
#define HK_G13_MESH_MAX_VERTICES           256u
#define HK_G13_MESH_MAX_PRIMITIVES         256u
#define HK_G13_MESH_MAX_OUTPUT_COMPONENTS 128u
#define HK_G13_MESH_MAX_SHARED_SIZE        (28u * 1024u)
#define HK_G13_MESH_MAX_OUTPUT_SIZE        (48u * 1024u)
/* Command-owned scratch is currently materialized for the complete direct
 * grid.  Bound the aggregate, not only its individually 32-bit-addressed
 * regions, so a legal large grid cannot turn six modest allocations into a
 * multi-gigabyte recorder stall.  256 MiB leaves hundreds of worst observed
 * title-profile groups while GPU-side dynamic consumption is unfinished. */
#define HK_G13_MESH_MAX_SCRATCH_SIZE (UINT64_C(256) * 1024 * 1024)

static inline bool
hk_g13_software_mesh_supported(enum agx_chip chip)
{
   return chip == AGX_CHIP_G13G || chip == AGX_CHIP_G13X;
}

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
   uint64_t primitive_records_size_B;
   uint64_t expanded_records_size_B;
   uint64_t index_buffer_size_B;
   uint64_t group_records_size_B;
   uint64_t draw_params_size_B;
   uint64_t scratch_size_B;
   uint32_t draw_count;
   uint32_t draw_stride_B;
};

bool hk_mesh_emulation_direct_requirements(
   enum agx_chip chip, const struct hk_api_shader *api,
   const uint32_t group_count[3],
   struct hk_mesh_emulation_direct_plan *plan);

bool hk_mesh_emulation_direct_plan_init(
   enum agx_chip chip, const struct hk_api_shader *api,
   const uint32_t group_count[3], uint64_t vertex_records,
   uint64_t vertex_records_size_B, uint64_t primitive_records,
   uint64_t primitive_records_size_B, uint64_t expanded_records,
   uint64_t expanded_records_size_B, uint64_t index_buffer,
   uint64_t index_buffer_size_B, uint64_t group_records,
   uint64_t group_records_size_B, uint64_t draw_params,
   uint64_t draw_params_size_B,
   struct hk_mesh_emulation_direct_plan *plan);
