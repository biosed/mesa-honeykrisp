/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_mesh_emulation_direct.h"

#include <limits.h>
#include <string.h>

#include "hk_shader.h"

static bool
add_scratch_size(uint64_t *total, uint64_t size)
{
   if (size > UINT64_MAX - *total)
      return false;

   *total += size;
   return true;
}

bool
hk_mesh_emulation_direct_requirements(
   enum agx_chip chip, const struct hk_api_shader *api,
   const uint32_t group_count[3],
   struct hk_mesh_emulation_direct_plan *plan)
{
   if (!api || !group_count || !plan ||
       !hk_g13_software_mesh_supported(chip) ||
       api->vk.stage != MESA_SHADER_MESH)
      return false;

   const struct hk_shader *producer =
      &api->variants[HK_MESH_VARIANT_MAIN];
   const struct hk_shader *raster =
      &api->variants[HK_MESH_VARIANT_RAST];
   const struct hk_shader *predicated =
      &api->variants[HK_MESH_VARIANT_PREDICATED_OBJECT];
   if (!producer->compiled || !producer->only_linked ||
       producer->b.info.stage != MESA_SHADER_COMPUTE ||
       !raster->compiled ||
       (!raster->linked.ht && !raster->linked.lazy_ht) ||
       raster->b.info.stage != MESA_SHADER_VERTEX || predicated->compiled ||
       !producer->info.mesh.max_vertices ||
       !producer->info.mesh.max_primitives ||
       producer->info.mesh.prim != MESA_PRIM_TRIANGLES ||
       !producer->info.mesh.vertex_stride_B ||
       !producer->info.mesh.primitive_stride_B ||
       !raster->info.mesh.vertex_stride_B)
      return false;

   uint64_t groups = 1;
   uint32_t threads[3];
   for (unsigned i = 0; i < 3; ++i) {
      const uint32_t local = producer->info.mesh.local_size[i];
      if (!group_count[i] || !local ||
          group_count[i] > UINT32_MAX / local ||
          groups > UINT64_MAX / group_count[i])
         return false;

      threads[i] = group_count[i] * local;
      groups *= group_count[i];
   }

   /* Command recording currently emits one ordinary indexed consumer per
    * group. Keep that host loop bounded until it is replaced by a GPU-side
    * indirect/count consumer. */
   if (groups > HK_G13_MESH_MAX_GROUP_TOTAL)
      return false;

   const uint64_t vertices =
      groups * producer->info.mesh.max_vertices;
   const uint64_t primitives =
      groups * producer->info.mesh.max_primitives;
   if (groups &&
       (vertices / groups != producer->info.mesh.max_vertices ||
        primitives / groups != producer->info.mesh.max_primitives))
      return false;

   if (vertices > UINT64_MAX / producer->info.mesh.vertex_stride_B ||
       primitives > UINT64_MAX / producer->info.mesh.primitive_stride_B ||
       primitives > UINT64_MAX / 3)
      return false;
   const uint64_t expanded_vertices = primitives * 3;
   if (expanded_vertices >
          UINT64_MAX / raster->info.mesh.vertex_stride_B ||
       expanded_vertices > UINT32_MAX ||
       groups > UINT64_MAX / sizeof(struct hk_mesh_emulation_group_record) ||
       groups > UINT64_MAX / sizeof(VkDrawIndexedIndirectCommand) ||
       groups > UINT32_MAX)
      return false;

   const uint64_t vertex_need =
      vertices * producer->info.mesh.vertex_stride_B;
   const uint64_t primitive_need =
      primitives * producer->info.mesh.primitive_stride_B;
   const uint64_t expanded_need =
      expanded_vertices * raster->info.mesh.vertex_stride_B;
   const uint64_t index_need =
      expanded_vertices * sizeof(uint32_t);
   const uint64_t group_need =
      groups * sizeof(struct hk_mesh_emulation_group_record);
   const uint64_t draw_need =
      groups * sizeof(VkDrawIndexedIndirectCommand);

   uint64_t scratch_need = sizeof(struct hk_mesh_emulation_params);
   if (!add_scratch_size(&scratch_need, vertex_need) ||
       !add_scratch_size(&scratch_need, primitive_need) ||
       !add_scratch_size(&scratch_need, expanded_need) ||
       !add_scratch_size(&scratch_need, index_need) ||
       !add_scratch_size(&scratch_need, group_need) ||
       !add_scratch_size(&scratch_need, draw_need))
      return false;

   if (scratch_need > HK_G13_MESH_MAX_SCRATCH_SIZE)
      return false;

   memset(plan, 0, sizeof(*plan));
   plan->producer = producer;
   plan->raster = raster;
   plan->producer_grid = agx_3d(threads[0], threads[1], threads[2]);
   plan->producer_local_size = agx_workgroup(
      producer->info.mesh.local_size[0], producer->info.mesh.local_size[1],
      producer->info.mesh.local_size[2]);
   plan->barrier = AGX_BARRIER_ALL | AGX_PREGFX;
   plan->vertex_records_size_B = vertex_need;
   plan->primitive_records_size_B = primitive_need;
   plan->expanded_records_size_B = expanded_need;
   plan->index_buffer_size_B = index_need;
   plan->group_records_size_B = group_need;
   plan->draw_params_size_B = draw_need;
   plan->scratch_size_B = scratch_need;
   plan->draw_count = groups;
   plan->draw_stride_B = sizeof(VkDrawIndexedIndirectCommand);
   return true;
}

bool
hk_mesh_emulation_direct_plan_init(
   enum agx_chip chip, const struct hk_api_shader *api,
   const uint32_t group_count[3], uint64_t vertex_records,
   uint64_t vertex_records_size_B, uint64_t primitive_records,
   uint64_t primitive_records_size_B, uint64_t expanded_records,
   uint64_t expanded_records_size_B, uint64_t index_buffer,
   uint64_t index_buffer_size_B, uint64_t group_records,
   uint64_t group_records_size_B, uint64_t draw_params,
   uint64_t draw_params_size_B,
   struct hk_mesh_emulation_direct_plan *plan)
{
   struct hk_mesh_emulation_direct_plan requirements;
   if (!hk_mesh_emulation_direct_requirements(chip, api, group_count,
                                               &requirements))
      return false;

   if (!vertex_records || (vertex_records & 15) || !primitive_records ||
       (primitive_records & 15) || !expanded_records ||
       (expanded_records & 15) || !index_buffer ||
       (index_buffer & 3) || !group_records || (group_records & 31) ||
       !draw_params || (draw_params & 3) ||
       requirements.vertex_records_size_B > vertex_records_size_B ||
       requirements.primitive_records_size_B > primitive_records_size_B ||
       requirements.expanded_records_size_B > expanded_records_size_B ||
       requirements.index_buffer_size_B > index_buffer_size_B ||
       index_buffer_size_B > UINT32_MAX ||
       requirements.group_records_size_B > group_records_size_B ||
       requirements.draw_params_size_B > draw_params_size_B)
      return false;

   *plan = requirements;
   plan->params = (struct hk_mesh_emulation_params){
      .vertex_records = vertex_records,
      .primitive_records = primitive_records,
      .expanded_records = expanded_records,
      .index_buffer = index_buffer,
      .group_records = group_records,
      .draw_params = draw_params,
      .mesh_grid = {group_count[0], group_count[1], group_count[2]},
      .abi_version = HK_MESH_EMULATION_ABI_VERSION,
      .flags = HK_MESH_EMULATION_FLAG_DEINDEXED,
   };
   plan->draw = agx_draw_indexed_indirect(
      draw_params, index_buffer, (uint32_t)index_buffer_size_B,
      AGX_INDEX_SIZE_U32, false);
   return true;
}
