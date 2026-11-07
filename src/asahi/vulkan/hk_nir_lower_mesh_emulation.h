/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "compiler/nir/nir.h"
#include "util/mesa-blake3.h"

/*
 * Driver-internal parameter block for a software mesh producer.  The command
 * path will place its address in draw.geometry_params before dispatching the
 * lowered shader.  Keep this ABI independent of Vulkan application memory.
 */
#define HK_MESH_EMULATION_ABI_VERSION 4u
#define HK_MESH_EMULATION_PARAMS_SIZE 128u
#define HK_MESH_EMULATION_TARGET_G13 13u
#define HK_MESH_EMULATION_MAX_SLOTS 64u
#define HK_MESH_EMULATION_FLAG_DEINDEXED UINT32_C(1)
#define HK_MESH_EMULATION_MAX_WORKGROUP_X 1024u
#define HK_MESH_EMULATION_MAX_WORKGROUP_Y 1024u
#define HK_MESH_EMULATION_MAX_WORKGROUP_Z 64u
#define HK_MESH_EMULATION_MAX_WORKGROUP_INVOCATIONS 1024u
#define HK_MESH_EMULATION_MAX_TASK_PAYLOAD (4u * 1024u)
#define HK_MESH_EMULATION_TASK_RECORD_HEADER_SIZE 16u
#define HK_MESH_EMULATION_MAX_TASK_GROUPS 4096u
#define HK_MESH_EMULATION_MAX_EMITTED_GROUPS 2048u
#define HK_MESH_EMULATION_FLAG_TASK UINT32_C(2)

struct hk_mesh_emulation_params {
   uint64_t vertex_records;
   uint64_t primitive_records;
   uint64_t expanded_records;
   uint64_t index_buffer;
   uint64_t group_records;
   uint64_t task_records;
   uint64_t draw_params;
   uint64_t group_to_task;
   uint32_t mesh_grid[3];
   uint32_t task_grid[3];
   uint32_t task_record_stride_B;
   uint32_t task_payload_size_B;
   uint32_t abi_version;
   uint32_t flags;
   uint32_t task_emitted_groups;
   uint32_t task_overflow;
   uint32_t reserved[4];
};

struct hk_mesh_emulation_task_record {
   uint32_t mesh_grid[3];
   uint32_t first_mesh_group;
};

struct hk_mesh_emulation_group_task {
   uint32_t task_index;
   uint32_t local_mesh_group;
};

struct hk_mesh_emulation_group_record {
   uint32_t expanded_vertex_count;
   uint32_t primitive_count;
   uint32_t expanded_vertex_base;
   uint32_t primitive_base;
   uint32_t task_index;
   uint32_t flags;
   uint32_t reserved[2];
};

struct hk_mesh_emulation_layout {
   unsigned vertex_count;
   unsigned primitive_count;
   unsigned shared_size;
   unsigned vertices_per_group;
   unsigned primitives_per_group;
   unsigned vertex_stride_B;
   unsigned primitive_stride_B;
   unsigned expanded_stride_B;
   unsigned position_offset_B;
   unsigned index_offset_B;
   unsigned output_memory_B;
   unsigned task_payload_size_B;
   uint64_t vertex_outputs;
   uint64_t primitive_outputs;
   uint16_t vertex_offset_B[HK_MESH_EMULATION_MAX_SLOTS];
   uint16_t primitive_offset_B[HK_MESH_EMULATION_MAX_SLOTS];
   uint16_t expanded_offset_B[HK_MESH_EMULATION_MAX_SLOTS];
   uint8_t vertex_components[HK_MESH_EMULATION_MAX_SLOTS];
   uint8_t primitive_components[HK_MESH_EMULATION_MAX_SLOTS];
   nir_alu_type output_type[HK_MESH_EMULATION_MAX_SLOTS];
};

/*
 * Immutable, cache-addressable result of admitting one API mesh shader to
 * the bounded G13 software contract.  It owns serialized NIR for both halves
 * of the pair so pipeline compilation can make private mutable clones and
 * still pass them through the ordinary Honeykrisp compiler.  In particular,
 * this object is not an uploaded shader and must never be submitted directly.
 */
struct hk_mesh_emulation_admission;

/*
 * Lower the bounded triangle-only mesh producer contract to a compute shader.
 *
 * The extension remains unadvertised on G13 until the command and generated
 * pull-vertex consumer paths are complete.  Returning false is fail-closed:
 * unsupported shaders are not modified.
 */
bool hk_nir_lower_mesh_emulation(
   nir_shader *shader, struct hk_mesh_emulation_layout *layout_out);

struct hk_task_emulation_layout {
   unsigned payload_size_B;
   unsigned task_record_stride_B;
   unsigned local_size[3];
};

/* Lower one API task workgroup to a bounded compute producer.  EmitMeshTasks
 * allocates a contiguous child range in GPU memory and publishes the
 * task/payload mapping consumed by hk_nir_lower_mesh_emulation(). */
bool hk_nir_lower_task_emulation(
   nir_shader *shader, struct hk_task_emulation_layout *layout_out);

/*
 * Build the ordinary vertex-stage consumer for the position scratch.  The
 * matching U32 index scratch is consumed by the fixed-function indexed draw;
 * its fetched index arrives here as load_vertex_id.
 */
nir_shader *hk_nir_mesh_emulation_pull_vs(
   const nir_shader_compiler_options *options,
   const struct hk_mesh_emulation_layout *layout);

/*
 * Build an immutable producer/pull-VS pair.  The identity covers the contract
 * version and both serialized NIR programs.  Unsupported input returns NULL
 * without changing the caller's shader.
 */
struct hk_mesh_emulation_admission *
hk_mesh_emulation_admission_create(
   const nir_shader *mesh, const nir_shader_compiler_options *options);

struct hk_mesh_emulation_admission *
hk_mesh_emulation_admission_ref(struct hk_mesh_emulation_admission *admission);

/* Returns true when this call released the final reference. */
bool hk_mesh_emulation_admission_unref(
   struct hk_mesh_emulation_admission *admission);

const uint8_t *hk_mesh_emulation_admission_identity(
   const struct hk_mesh_emulation_admission *admission);

const struct hk_mesh_emulation_layout *hk_mesh_emulation_admission_layout(
   const struct hk_mesh_emulation_admission *admission);

nir_shader *hk_mesh_emulation_admission_clone_producer(
   const struct hk_mesh_emulation_admission *admission,
   const nir_shader_compiler_options *options);

nir_shader *hk_mesh_emulation_admission_clone_pull_vs(
   const struct hk_mesh_emulation_admission *admission,
   const nir_shader_compiler_options *options);
