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
#define HK_MESH_EMULATION_ABI_VERSION 5u
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
/* The current G13 software path records and retains one raster consumer plus
 * worst-case scratch for every possible child.  Keep this experimental limit
 * coherent in task lowering, allocation, predicate and advertised limits.
 * The first title task pair emits at most 32 children per input group (64 for
 * its direct count of two); larger emissions fail closed through task_overflow
 * instead of producing a partial draw. */
#define HK_MESH_EMULATION_MAX_EMITTED_GROUPS 64u
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
   /* Mesh output topology (MESA_PRIM_POINTS, _LINES or _TRIANGLES) and the
    * matching number of PrimitiveIndices words per primitive (1, 2 or 3). */
   unsigned primitive_type;
   unsigned indices_per_primitive;
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

/* Index words per mesh primitive for an admitted output topology, or 0 for
 * a topology outside the software contract. */
static inline unsigned
hk_mesh_emulation_indices_per_primitive(unsigned primitive_type)
{
   switch (primitive_type) {
   case MESA_PRIM_POINTS:
      return 1;
   case MESA_PRIM_LINES:
      return 2;
   case MESA_PRIM_TRIANGLES:
      return 3;
   default:
      return 0;
   }
}

/* Whether the producer copies every primitive's vertices into expanded
 * records (deindexed) instead of indexing the source vertex records.
 * Per-primitive user outputs need a copy per primitive.  Point and line
 * meshes always take the copy so that an inactive or invalid primitive can be
 * given a position outside the clip volume: unlike a triangle, a point or
 * line with repeated indices is not a zero-area primitive. */
static inline bool
hk_mesh_emulation_layout_expanded(const struct hk_mesh_emulation_layout *l)
{
   return (l->primitive_outputs & ~VARYING_BIT_PRIMITIVE_INDICES) ||
          l->primitive_type != MESA_PRIM_TRIANGLES;
}

/*
 * Lower the bounded point/line/triangle mesh producer contract to a compute
 * shader.  PointSize is carried per vertex for point meshes only.
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

/* Preserve Vulkan TaskEXT NumWorkgroups before generic preprocessing lowers
 * it to a graphics-root pointer.  The G13 task producer instead owns the
 * command parameters' task_grid field. */
bool hk_nir_lower_g13_task_num_workgroups(nir_shader *shader);

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

/* Bound indirect task-group allocation via ASAHI_G13_SOFTWARE_TASK_INDIRECT_CAP
 * (defaulting to HK_MESH_EMULATION_MAX_TASK_GROUPS = 4096 when unset). */
static inline uint32_t
hk_mesh_emulation_indirect_task_cap(int64_t requested_cap)
{
   if (requested_cap < 1)
      return 1u;
   if (requested_cap > (int64_t)HK_MESH_EMULATION_MAX_TASK_GROUPS)
      return HK_MESH_EMULATION_MAX_TASK_GROUPS;
   return (uint32_t)requested_cap;
}

struct hk_mesh_emulation_scratch_offsets {
   uint64_t vertex_records_offset_B;
   uint64_t primitive_records_offset_B;
   uint64_t expanded_records_offset_B;
   uint64_t index_buffer_offset_B;
   uint64_t group_records_offset_B;
   uint64_t task_records_offset_B;
   uint64_t group_to_task_offset_B;
   uint64_t draw_params_offset_B;
   uint64_t patched_draw_params_offset_B;
   uint64_t total_size_B;
};

static inline bool
hk_mesh_emulation_align_add_u64(uint64_t *cursor, uint64_t size_B,
                                uint64_t align_B, uint64_t *offset_out)
{
   const uint64_t mask = align_B - 1;
   if (align_B == 0 || (align_B & mask) != 0 || *cursor > UINT64_MAX - mask)
      return false;
   const uint64_t aligned = (*cursor + mask) & ~mask;
   if (size_B > UINT64_MAX - aligned)
      return false;
   *offset_out = aligned;
   *cursor = aligned + size_B;
   return true;
}

/* Pack the six (taskless) or nine (task+mesh) scratch regions of a software
 * mesh draw into a single 32-byte-aligned allocation so each active slot uses
 * one BO instead of 4-5 separate large_bos allocations. */
static inline bool
hk_mesh_emulation_pack_scratch_offsets(
   uint64_t vertex_records_size_B, uint64_t primitive_records_size_B,
   uint64_t expanded_records_size_B, uint64_t index_buffer_size_B,
   uint64_t group_records_size_B, uint64_t task_records_size_B,
   uint64_t group_to_task_size_B, uint64_t draw_params_size_B,
   uint64_t patched_draw_params_size_B, bool task, uint64_t max_scratch_B,
   struct hk_mesh_emulation_scratch_offsets *out)
{
   if (!out || !vertex_records_size_B || !primitive_records_size_B ||
       !expanded_records_size_B || !index_buffer_size_B ||
       !group_records_size_B || !draw_params_size_B ||
       (task && (!task_records_size_B || !group_to_task_size_B ||
                 !patched_draw_params_size_B)))
      return false;

   memset(out, 0, sizeof(*out));
   uint64_t cursor = 0;
   uint64_t dummy = 0;
   if (!hk_mesh_emulation_align_add_u64(
          &cursor, group_records_size_B, 32, &out->group_records_offset_B) ||
       !hk_mesh_emulation_align_add_u64(
          &cursor, vertex_records_size_B, 16, &out->vertex_records_offset_B) ||
       !hk_mesh_emulation_align_add_u64(
          &cursor, primitive_records_size_B, 16,
          &out->primitive_records_offset_B) ||
       !hk_mesh_emulation_align_add_u64(
          &cursor, expanded_records_size_B, 16,
          &out->expanded_records_offset_B) ||
       (task &&
        !hk_mesh_emulation_align_add_u64(
           &cursor, task_records_size_B, 16, &out->task_records_offset_B)) ||
       (task &&
        !hk_mesh_emulation_align_add_u64(
           &cursor, group_to_task_size_B, 8, &out->group_to_task_offset_B)) ||
       !hk_mesh_emulation_align_add_u64(
          &cursor, index_buffer_size_B, 4, &out->index_buffer_offset_B) ||
       !hk_mesh_emulation_align_add_u64(
          &cursor, draw_params_size_B, 4, &out->draw_params_offset_B) ||
       (task &&
        !hk_mesh_emulation_align_add_u64(
           &cursor, patched_draw_params_size_B, 4,
           &out->patched_draw_params_offset_B)) ||
       !hk_mesh_emulation_align_add_u64(&cursor, 0, 32, &dummy))
      return false;

   if (cursor == 0 || cursor > UINT32_MAX || cursor > max_scratch_B)
      return false;

   out->total_size_B = cursor;
   return true;
}

#define HK_G13_MESH_SCRATCH_POOL_SLOTS 16u

struct hk_cs;

struct hk_mesh_emulation_scratch_slot {
   void *cpu;
   uint64_t gpu;
   uint32_t capacity_B;
   bool busy_in_pass;
   struct hk_cs *last_gfx_cs;
};

struct hk_mesh_emulation_scratch_pool {
   struct hk_mesh_emulation_scratch_slot
      slots[HK_G13_MESH_SCRATCH_POOL_SLOTS];
   uint32_t count;
};

static inline void
hk_mesh_emulation_scratch_pool_reset(
   struct hk_mesh_emulation_scratch_pool *pool)
{
   memset(pool, 0, sizeof(*pool));
}

static inline void
hk_mesh_emulation_scratch_pool_end_pass(
   struct hk_mesh_emulation_scratch_pool *pool, struct hk_cs *gfx_cs)
{
   for (uint32_t i = 0; i < pool->count; ++i) {
      if (pool->slots[i].busy_in_pass) {
         pool->slots[i].busy_in_pass = false;
         if (gfx_cs)
            pool->slots[i].last_gfx_cs = gfx_cs;
      }
   }
}

static inline bool
hk_mesh_emulation_scratch_pool_acquire(
   struct hk_mesh_emulation_scratch_pool *pool, uint32_t needed_B,
   void **cpu_out, uint64_t *gpu_out, struct hk_cs **wait_gfx_cs_out)
{
   uint32_t best = UINT32_MAX;
   for (uint32_t i = 0; i < pool->count; ++i) {
      const struct hk_mesh_emulation_scratch_slot *s = &pool->slots[i];
      if (s->busy_in_pass || s->capacity_B < needed_B)
         continue;
      if (best == UINT32_MAX || s->capacity_B < pool->slots[best].capacity_B)
         best = i;
   }
   if (best == UINT32_MAX)
      return false;

   struct hk_mesh_emulation_scratch_slot *s = &pool->slots[best];
   s->busy_in_pass = true;
   *cpu_out = s->cpu;
   *gpu_out = s->gpu;
   *wait_gfx_cs_out = s->last_gfx_cs;
   s->last_gfx_cs = NULL;
   return true;
}

static inline void
hk_mesh_emulation_scratch_pool_insert(
   struct hk_mesh_emulation_scratch_pool *pool, void *cpu, uint64_t gpu,
   uint32_t capacity_B)
{
   uint32_t target = pool->count < HK_G13_MESH_SCRATCH_POOL_SLOTS
                        ? pool->count++
                        : UINT32_MAX;
   if (target == UINT32_MAX) {
      for (uint32_t i = 0; i < pool->count; ++i) {
         if (!pool->slots[i].busy_in_pass &&
             pool->slots[i].capacity_B < capacity_B &&
             (target == UINT32_MAX ||
              pool->slots[i].capacity_B < pool->slots[target].capacity_B))
            target = i;
      }
   }
   if (target < HK_G13_MESH_SCRATCH_POOL_SLOTS) {
      pool->slots[target] = (struct hk_mesh_emulation_scratch_slot){
         .cpu = cpu,
         .gpu = gpu,
         .capacity_B = capacity_B,
         .busy_in_pass = true,
         .last_gfx_cs = NULL,
      };
   }
}
